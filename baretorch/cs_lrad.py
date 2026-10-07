# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/baretorch/cs_lrad.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


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
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class LowRankAssociativeDeltaEngine(nn.Module):
    def __init__(self, d_model=256, num_heads=16, chunk_size=32, rank=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.r = rank

        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head

        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)

        self.W_u = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        self.W_r = nn.Linear(d_model, self.num_heads * self.r, bias=False)

        self.W_gate = nn.Linear(d_model, num_heads, bias=True)
        self.W_beta_gate = nn.Linear(d_model, num_heads, bias=True)

        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        H, C, d_h, r = self.num_heads, self.chunk_size, self.d_head, self.r
        scaling = 1.0 / math.sqrt(d_h)

        N = (L + C - 1) // C
        pad_len = (C - (L % C)) % C

        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len), value=0)
        else:
            x_padded = x

        if attention_mask is not None:
            if pad_len > 0:
                mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
            else:
                mask_padded = attention_mask
            mask_5d = mask_padded.view(B, N, C).unsqueeze(1).unsqueeze(-1).to(dtype=torch.float32, device=x.device)
        else:
            seq_idx = torch.arange(N * C, device=x.device)
            mask_padded = (seq_idx < L).to(dtype=torch.float32).view(1, N, C)
            mask_5d = mask_padded.unsqueeze(1).unsqueeze(-1)

        Q = F.silu(self.W_q(x_padded).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        K = F.silu(self.W_k(x_padded).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        V = self.W_v(x_padded).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)

        U = self.W_u(x_padded).view(B, N, C, H, r).permute(0, 3, 1, 2, 4)
        R = self.W_r(x_padded).view(B, N, C, H, r).permute(0, 3, 1, 2, 4)

        gate_fp32 = torch.clamp(
            torch.sigmoid(self.W_gate(x_padded).float()).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1),
            min=1e-4,
            max=0.9999,
        )
        beta_gate_fp32 = torch.sigmoid(self.W_beta_gate(x_padded).float()).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)

        U_fp32 = U.float() * mask_5d
        V_fp32 = V.float() * mask_5d
        gate_fp32 = torch.where(mask_5d.bool(), gate_fp32, torch.ones_like(gate_fp32))

        log_gate = torch.log(gate_fp32)
        Lambda = torch.cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(torch.clamp(Lambda, min=-30.0, max=0.0))

        indices = torch.arange(C, device=x.device)
        causal_mask = (indices[:, None] >= indices[None, :]).view(1, 1, 1, C, C)

        diff = Lambda - Lambda.transpose(-1, -2)
        diff_masked = torch.where(causal_mask, diff, torch.tensor(-10000.0, device=x.device, dtype=torch.float32))
        M_links = torch.exp(torch.clamp(diff_masked, max=0.0))

        Y_local = torch.matmul(torch.matmul(Q.float(), K.float().transpose(-1, -2)) * scaling * M_links, V_fp32)

        chunk_decay_log = torch.clamp(torch.sum(log_gate, dim=-2).squeeze(-1), min=-30.0, max=0.0)
        Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)

        c_indices = torch.arange(N, device=x.device)
        causal_mask_chunks = (c_indices[:, None] > c_indices[None, :]).view(1, 1, N, N)

        log_M_chunks_masked = torch.where(
            causal_mask_chunks,
            log_M_chunks,
            torch.tensor(-10000.0, device=x.device, dtype=torch.float32),
        )
        M_chunks = torch.exp(torch.clamp(log_M_chunks_masked, max=0.0))

        decay = torch.exp(torch.clamp(Lambda[:, :, :, -1:, :] - Lambda, min=-30.0, max=0.0))
        U_decayed = (U_fp32 * beta_gate_fp32) * decay

        S_historical_flat = torch.matmul(
            M_chunks,
            torch.matmul(U_decayed.transpose(-1, -2), V_fp32).view(B, H, N, r * d_h),
        )
        S_historical = S_historical_flat.view(B, H, N, r, d_h)

        Y_global = torch.matmul(R.float() * exp_Lambda, S_historical) * scaling

        Out = (Y_local + Y_global).permute(0, 2, 3, 1, 4).contiguous().view(B, N * C, self.inner_dim).to(dtype=x.dtype)
        Out = Out[:, :L, :]

        output = self.resid_drop(self.W_out(Out * F.silu(self.W_swish_gate(x))))

        chunk_decay_last = exp_Lambda[:, :, -1, -1:, :]
        S_historical_last = S_historical[:, :, -1]
        S_local_last = torch.matmul(U_decayed[:, :, -1].transpose(-1, -2), V_fp32[:, :, -1])
        S_final = (chunk_decay_last * S_historical_last) + S_local_last
        S_final = torch.clamp(S_final, min=-50.0, max=50.0).to(dtype=x.dtype)

        return output, S_final

    def step_inference(self, x, past_S=None, attention_mask=None):
        B, L, D = x.shape
        H, d_h, r = self.num_heads, self.d_head, self.r

        Q = F.silu(self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K = F.silu(self.W_k(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        V = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        U = self.W_u(x).view(B, L, H, r).permute(0, 2, 1, 3)
        R = self.W_r(x).view(B, L, H, r).permute(0, 2, 1, 3)

        gate = torch.clamp(
            torch.sigmoid(self.W_gate(x).float()).view(B, L, H).permute(0, 2, 1).unsqueeze(-1),
            min=1e-4,
            max=0.9999,
        )
        beta_gate = torch.sigmoid(self.W_beta_gate(x).float()).view(B, L, H).permute(0, 2, 1).unsqueeze(-1)

        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.size(1) != L:
                attn_mask_curr = attention_mask[:, -L:]
            else:
                attn_mask_curr = attention_mask
            mask = attn_mask_curr.view(B, 1, L, 1).to(dtype=torch.float32, device=x.device)
            U_fp32 = U.float() * mask
            V_fp32 = V.float() * mask
            gate = torch.where(mask.bool(), gate, torch.ones_like(gate))
        else:
            U_fp32 = U.float()
            V_fp32 = V.float()

        if past_S is None or not isinstance(past_S, torch.Tensor):
            past_S = torch.zeros(B, H, r, d_h, device=x.device, dtype=torch.float32)
        else:
            past_S = past_S.float()

        Y_global = torch.matmul(R.float(), past_S) * (1.0 / math.sqrt(d_h))
        Y_local = torch.matmul(Q.float(), torch.matmul(K.float().transpose(-1, -2), V_fp32)) * (1.0 / math.sqrt(d_h))

        S_state = (gate * past_S) + torch.matmul((U_fp32 * beta_gate).transpose(-1, -2), V_fp32)
        S_state = torch.clamp(S_state, min=-50.0, max=50.0)

        Out = (Y_local + Y_global).permute(0, 2, 1, 3).contiguous().view(B, L, D).to(dtype=x.dtype)
        output = self.resid_drop(self.W_out(Out * F.silu(self.W_swish_gate(x))))
        return output, S_state.to(dtype=x.dtype)


class LRADDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, rank=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = LowRankAssociativeDeltaEngine(d_model, num_heads, chunk_size=chunk_size, rank=rank, dropout=dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x, past_state=None, use_cache=False, attention_mask=None):
        h_attn = self.ln1(x)
        B, L, _ = x.shape

        if isinstance(past_state, tuple) and len(past_state) > 0:
            past_state = past_state[0]

        is_step_inference = (past_state is not None) or (L == 1)

        if is_step_inference:
            attn_out, next_state = self.attn.step_inference(h_attn, past_S=past_state, attention_mask=attention_mask)
        else:
            attn_out, next_state = self.attn(h_attn, attention_mask=attention_mask)

        x_out = x + attn_out
        x_out = x_out + self.mlp(self.ln2(x_out))

        cache_out = (next_state, next_state) if use_cache else None
        return x_out, cache_out


class CSLRADConfig(PretrainedConfig):
    model_type = "cs_lrad"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_heads=16,
        num_layers=8,
        chunk_size=32,
        rank=8,
        dropout=0.1,
        use_grad_checkpointing=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.rank = rank
        self.dropout = dropout
        self.use_grad_checkpointing = use_grad_checkpointing


class CSLRADPreTrainedModel(PreTrainedModel):
    config_class = CSLRADConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CSLRADModel(CSLRADPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            LRADDecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                chunk_size=config.chunk_size,
                rank=config.rank,
                dropout=config.dropout,
                use_grad_checkpointing=config.use_grad_checkpointing,
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
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
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        h = inputs_embeds
        next_decoder_cache = [] if use_cache else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)

            past_state = past_key_values[i] if past_key_values is not None else None
            h, next_state = layer(
                h,
                past_state=past_state,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )

            if use_cache:
                next_decoder_cache.append(next_state)

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


class CSLRADForCausalLM(CSLRADPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = CSLRADModel(config)
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

        # Logit Soft-Capping (30.0 * tanh(logits / 30.0))
        logits = 30.0 * torch.tanh(logits / 30.0)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past += (None,)
            else:
                reordered_past += (layer_past.index_select(0, beam_idx),)
        return reordered_past