# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/baretorch/transformer.py
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.to(torch.float32)).to(input_dtype)


class GatedMLP(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff if d_ff is not None else int(d_model * 3.5)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device="cpu", dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if getattr(self.inv_freq, "is_meta", False) or self.inv_freq.device.type == "meta":
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        else:
            inv_freq = self.inv_freq.to(device)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, q, k, position_ids):
        if position_ids is not None and position_ids.shape[-1] != q.shape[-2]:
            if q.shape[-2] == 1:
                position_ids = position_ids[:, -1:]
            else:
                position_ids = position_ids[:, -q.shape[-2] :]

        max_pos = int(position_ids.max().item()) + 1 if position_ids is not None else q.shape[-2]

        if (
            self.cos_cached is None
            or max_pos > self.cos_cached.size(0)
            or self.cos_cached.device != q.device
            or self.cos_cached.dtype != q.dtype
        ):
            new_len = max(max_pos + 1024, self.max_position_embeddings)
            self._set_cos_sin_cache(seq_len=new_len, device=q.device, dtype=q.dtype)

        position_ids = position_ids.to(q.device)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(dtype=q.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(dtype=q.dtype)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed.to(dtype=q.dtype), k_embed.to(dtype=k.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def unwrap_kv_pair(past_kv):
    if past_kv is None:
        return None, None
    p_item = past_kv
    while isinstance(p_item, (tuple, list)) and len(p_item) > 0 and not isinstance(p_item[0], torch.Tensor):
        p_item = p_item[0]
    if isinstance(p_item, (tuple, list)) and len(p_item) >= 2:
        pk, pv = p_item[0], p_item[1]
        if isinstance(pk, torch.Tensor) and isinstance(pv, torch.Tensor):
            return pk, pv
    return None, None


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads=16,
        num_kv_heads=4,
        dropout=0.0,
        max_seq_len=4096,
        use_qk_norm=False,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = d_model // num_heads
        self.use_qk_norm = use_qk_norm

        assert num_heads % self.num_kv_heads == 0, "Query heads must be divisible by KV heads."
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len, base=rope_theta)
        self.dropout_p = dropout
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, position_ids=None, attention_mask=None):
        B, L, D = x.shape
        H_q, H_kv, d_h = self.num_heads, self.num_kv_heads, self.head_dim

        pk_past, pv_past = unwrap_kv_pair(past_kv)

        if position_ids is None:
            past_len = pk_past.size(-2) if pk_past is not None else 0
            position_ids = torch.arange(past_len, past_len + L, dtype=torch.long, device=x.device).unsqueeze(0)

        q = self.W_q(x).view(B, L, H_q, d_h)
        k = self.W_k(x).view(B, L, H_kv, d_h)
        v = self.W_v(x).view(B, L, H_kv, d_h).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        q, k = self.rope.apply_rope(q, k, position_ids)

        if pk_past is not None and pv_past is not None:
            if pk_past.shape[-2] != pv_past.shape[-2]:
                min_len = min(pk_past.shape[-2], pv_past.shape[-2])
                pk_past = pk_past[..., :min_len, :]
                pv_past = pv_past[..., :min_len, :]
            k = torch.cat([pk_past, k], dim=-2)
            v = torch.cat([pv_past, v], dim=-2)

        current_kv = (k, v)

        if H_kv != H_q:
            k = repeat_kv(k, self.num_queries_per_kv)
            v = repeat_kv(v, self.num_queries_per_kv)

        is_causal_mask = (past_kv is None) and (L > 1)
        dropout_p = self.dropout_p if self.training else 0.0

        q = q.to(dtype=x.dtype)
        k = k.to(dtype=x.dtype)
        v = v.to(dtype=x.dtype)

        sdpa_mask = None
        if attention_mask is not None and is_causal_mask:
            # If all tokens are valid, maintain native FlashAttention causal execution
            if not attention_mask.all():
                causal_bool = torch.ones((L, L), dtype=torch.bool, device=q.device).tril()
                causal_bool = causal_bool.view(1, 1, L, L).expand(B, 1, L, L)
                pad_bool = attention_mask.bool().view(B, 1, 1, L)
                sdpa_mask = causal_bool & pad_bool
                is_causal_mask = False

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=is_causal_mask
        )

        out_flat = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_drop(self.W_out(out_flat)), current_kv


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads=4,
        d_ff=None,
        dropout=0.0,
        max_seq_len=4096,
        use_grad_checkpointing=False,
        use_qk_norm=False,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_qk_norm=use_qk_norm,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.ln2 = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = GatedMLP(d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x, past_kv=None, position_ids=None, attention_mask=None):
        def _block_forward(x_in, p_kv, pos_ids, attn_mask):
            attn_out, current_kv = self.attn(
                self.ln1(x_in), past_kv=p_kv, position_ids=pos_ids, attention_mask=attn_mask
            )
            x_out = x_in + attn_out
            x_out = x_out + self.mlp(self.ln2(x_out))
            return x_out, current_kv

        if self.use_grad_checkpointing and self.training:
            return checkpoint.checkpoint(
                _block_forward, x, past_kv, position_ids, attention_mask, use_reentrant=False
            )
        else:
            return _block_forward(x, past_kv, position_ids, attention_mask)


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_heads=16,
        num_kv_heads=4,
        num_layers=8,
        intermediate_size=None,
        dropout=0.0,
        max_seq_len=4096,
        use_grad_checkpointing=False,
        use_qk_norm=False,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        logit_soft_cap=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_grad_checkpointing = use_grad_checkpointing
        self.use_qk_norm = use_qk_norm
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.logit_soft_cap = logit_soft_cap


class TransformerPreTrainedModel(PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                d_ff=getattr(config, "intermediate_size", None),
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_grad_checkpointing=config.use_grad_checkpointing,
                use_qk_norm=getattr(config, "use_qk_norm", False),
                rope_theta=getattr(config, "rope_theta", 10000.0),
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        h = self.drop(inputs_embeds)
        next_decoder_cache = [] if use_cache else None

        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                for layer_past in past_key_values:
                    pk, _ = unwrap_kv_pair(layer_past)
                    if pk is not None:
                        past_length = pk.size(-2)
                        break

            if attention_mask is not None and past_length == 0:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)
            else:
                position_ids = torch.arange(
                    past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0)

        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)

            past_kv = past_key_values[i] if past_key_values is not None else None
            h, current_kv = layer(h, past_kv=past_kv, position_ids=position_ids, attention_mask=attention_mask)

            if use_cache:
                next_decoder_cache.append(current_kv)

        h = self.final_norm(h)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (h,)

        if not return_dict:
            return tuple(v for v in [h, next_decoder_cache, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )


class TransformerForCausalLM(TransformerPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        soft_cap = getattr(self.config, "logit_soft_cap", None)
        if soft_cap is not None and soft_cap > 0.0:
            logits = torch.clamp(logits, min=-float(soft_cap), max=float(soft_cap))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )