# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/baretorch/modeling_baretorch.py
from functools import partial
import inspect
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import AutoModel, AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_baretorch import BareTorchConfig, CSLRADConfig
from .cs_lrad import CSLRADForCausalLM, LowRankAssociativeDeltaEngine, LRADDecoderBlock, RMSNorm
from .transformer import (
    CausalSelfAttention,
    TransformerConfig,
    TransformerDecoderBlock,
    TransformerForCausalLM,
    unwrap_kv_pair,
)


def _extract_past_length(past_key_values, attention_mask=None, seq_length=1):
    """
    Extracts the number of previously cached tokens across hybrid layer caches:
      - CS-LRAD cache tuple: `(past_S, buf_k, buf_v, buf_u, buf_lambda, curr_seq_len)` -> `layer_past[5]`
      - HuggingFace `DynamicCache` (used by spliced Qwen3_5DecoderLayer blocks) -> `.get_seq_length()`
      - Native Transformer `(k, v)` tuple -> `pk.size(-2)`
    """
    if past_key_values is None:
        return 0

    for layer_past in past_key_values:
        if layer_past is None:
            continue
        if isinstance(layer_past, (tuple, list)) and len(layer_past) >= 6 and isinstance(layer_past[5], int):
            return layer_past[5]
        if hasattr(layer_past, "get_seq_length"):
            seq_len_cached = layer_past.get_seq_length()
            if seq_len_cached > 0:
                return int(seq_len_cached)
        pk, _ = unwrap_kv_pair(layer_past)
        if pk is not None:
            return int(pk.size(-2))

    if attention_mask is not None and attention_mask.size(1) > seq_length:
        return int(attention_mask.size(1) - seq_length)
    return 0


class BareTorchPreTrainedModel(PreTrainedModel):
    config_class = BareTorchConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_quantized_cache = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LowRankAssociativeDeltaEngine):
            module._init_gate_parameters()

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None, *args, **kwargs):
        if "value" in kwargs:
            enable = kwargs.pop("value")

        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = partial(checkpoint.checkpoint, use_reentrant=False)

        self.gradient_checkpointing = enable
        if hasattr(self, "config"):
            self.config.use_grad_checkpointing = enable

        for module in self.modules():
            if hasattr(module, "use_grad_checkpointing"):
                module.use_grad_checkpointing = enable
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable
            module._gradient_checkpointing_func = gradient_checkpointing_func
            if hasattr(module, "gradient_checkpointing") and module is not self and module is not getattr(self, "model", None):
                module.gradient_checkpointing = False

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        if "past_key_values" not in model_kwargs:
            model_kwargs["past_key_values"] = None


class BareTorchModel(BareTorchPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = partial(checkpoint.checkpoint, use_reentrant=False)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.rotary_emb = None
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        rms_norm_unit_offset = getattr(config, "rms_norm_unit_offset", False)

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_type = config.layer_types[i]
            if layer_type == "cs_lrad":
                block = LRADDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=getattr(config, "intermediate_size", None),
                    chunk_size=config.chunk_size,
                    rank=config.rank,
                    dropout=config.dropout,
                    rms_norm_eps=rms_norm_eps,
                    rms_norm_unit_offset=rms_norm_unit_offset,
                    use_grad_checkpointing=config.use_grad_checkpointing,
                )
            elif layer_type == "transformer":
                # Instantiates our native Transformer block
                block = TransformerDecoderBlock(
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
            elif layer_type == "identity":
                block = nn.Identity()
            else:
                raise ValueError(f"Unsupported layer type '{layer_type}' at index {i}")
            self.layers.append(block)

        self.final_norm = RMSNorm(config.d_model, eps=rms_norm_eps, unit_offset=rms_norm_unit_offset)
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
        return_dict = return_dict if return_dict is not None else bool(getattr(self.config, "return_dict", True))

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

        if past_key_values is not None and inputs_embeds.size(1) > 1 and not self.training:
            inputs_embeds = inputs_embeds[:, -1:, :]
            batch_size, seq_length, _ = inputs_embeds.shape

        h = self.drop(inputs_embeds)
        next_decoder_cache = [] if use_cache else None

        past_length = _extract_past_length(past_key_values, attention_mask=attention_mask, seq_length=seq_length)

        if position_ids is None:
            if attention_mask is not None and past_length == 0:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)
            else:
                position_ids = torch.arange(
                    past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape[-1] != seq_length:
            position_ids = position_ids[:, -seq_length:]

        cache_position = torch.arange(
            past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device
        )

        position_embeddings = None
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(h, position_ids)

        all_hidden_states = () if output_hidden_states else None
        is_gc = getattr(self, "gradient_checkpointing", False) and self.training and getattr(self, "_gradient_checkpointing_func", None) is not None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)

            past_state = past_key_values[i] if past_key_values is not None else None
            cls_name = layer.__class__.__name__.lower()

            # 1. Native Transformer Layer Dispatch
            if isinstance(layer, TransformerDecoderBlock) or ("transformer" in cls_name and "qwen" not in cls_name):
                if is_gc and not getattr(layer, "use_grad_checkpointing", False):
                    def create_transformer_forward(module, p_kv, pos_ids, attn_mask):
                        def transformer_forward(x_in):
                            return module(x_in, past_kv=p_kv, position_ids=pos_ids, attention_mask=attn_mask)
                        return transformer_forward

                    h, next_state = self._gradient_checkpointing_func(
                        create_transformer_forward(layer, past_state, position_ids, attention_mask),
                        h,
                    )
                else:
                    h, next_state = layer(h, past_kv=past_state, position_ids=position_ids, attention_mask=attention_mask)

            # 2. Native CS-LRAD Recurrent Layer Dispatch
            elif isinstance(layer, LRADDecoderBlock) or "lrad" in cls_name:
                if is_gc and not getattr(layer, "use_grad_checkpointing", False):
                    def create_lrad_forward(module, p_state, u_cache, attn_mask):
                        def lrad_forward(x_in):
                            return module(x_in, past_state=p_state, use_cache=u_cache, attention_mask=attn_mask)
                        return lrad_forward

                    h, next_state = self._gradient_checkpointing_func(
                        create_lrad_forward(layer, past_state, use_cache, attention_mask),
                        h,
                    )
                else:
                    h, next_state = layer(h, past_state=past_state, use_cache=use_cache, attention_mask=attention_mask)

            # 3. Identity Pass-Through
            elif isinstance(layer, nn.Identity) or cls_name == "identity":
                next_state = None

            # 4. Spliced External Teacher Layer (e.g. Qwen2DecoderLayer / Qwen3_5DecoderLayer)
            else:
                qwen_attn_mask = attention_mask
                if qwen_attn_mask is not None and qwen_attn_mask.dim() == 2:
                    if bool(qwen_attn_mask.all()):
                        qwen_attn_mask = None
                    else:
                        # Convert 2D padding mask [B, K_len] into 4D additive causal/padding mask [B, 1, Q_len, K_len]
                        k_len = qwen_attn_mask.size(1)
                        q_len = h.size(1)
                        min_val = torch.finfo(h.dtype).min
                        pad_mask = (1.0 - qwen_attn_mask[:, None, None, :].to(h.dtype)) * min_val
                        if q_len > 1:
                            causal = torch.full((q_len, k_len), min_val, device=h.device, dtype=h.dtype)
                            causal = torch.triu(causal, diagonal=1 + (k_len - q_len)).unsqueeze(0).unsqueeze(0)
                            qwen_attn_mask = torch.minimum(pad_mask, causal)
                        else:
                            qwen_attn_mask = pad_mask

                # Each spliced Qwen full-attn layer uses its own standalone DynamicCache (slot 0)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                    layer.self_attn.layer_idx = 0

                qwen_cache = None
                if use_cache:
                    qwen_cache = past_state if isinstance(past_state, DynamicCache) else DynamicCache()

                sig_params = inspect.signature(layer.forward).parameters
                qwen_kwargs = {
                    "attention_mask": qwen_attn_mask,
                    "position_ids": position_ids,
                    "output_attentions": False,
                    "use_cache": use_cache,
                    "position_embeddings": position_embeddings,
                }
                if "cache_position" in sig_params:
                    qwen_kwargs["cache_position"] = cache_position
                if qwen_cache is not None:
                    if "past_key_values" in sig_params:
                        qwen_kwargs["past_key_values"] = qwen_cache
                    elif "past_key_value" in sig_params:
                        qwen_kwargs["past_key_value"] = qwen_cache

                if is_gc:
                    def create_qwen_forward(module, kw, qc):
                        def qwen_forward(x_in):
                            layer_outputs = module(x_in, **kw)
                            if isinstance(layer_outputs, tuple):
                                h_res = layer_outputs[0]
                            elif isinstance(layer_outputs, torch.Tensor):
                                h_res = layer_outputs
                            else:
                                h_res = getattr(layer_outputs, "last_hidden_state", layer_outputs[0])
                            return h_res, qc
                        return qwen_forward

                    h, next_state = self._gradient_checkpointing_func(
                        create_qwen_forward(layer, qwen_kwargs, qwen_cache),
                        h,
                    )
                else:
                    layer_outputs = layer(h, **qwen_kwargs)
                    if isinstance(layer_outputs, tuple):
                        h = layer_outputs[0]
                    elif isinstance(layer_outputs, torch.Tensor):
                        h = layer_outputs
                    else:
                        h = getattr(layer_outputs, "last_hidden_state", layer_outputs[0])
                    next_state = qwen_cache if use_cache else None

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


class BareTorchForCausalLM(BareTorchPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}
    supports_gradient_checkpointing = True
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_quantized_cache = False

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = partial(checkpoint.checkpoint, use_reentrant=False)
        self.model = BareTorchModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._is_cs_lrad_compiled = False
        self.post_init()

    def compile_cs_lrad(self, mode: str = "default", dynamic: bool = True):
        """
        Compiles all CS-LRAD recurrent blocks (`chunk_forward`, `ln1`, `ln2`, `mlp`) using
        `torch.compile(..., dynamic=True)` so arbitrary prompt lengths and batch sizes run at
        peak fused Triton/Inductor speed without shape-recompilation stalls.
        """
        if getattr(self, "_is_cs_lrad_compiled", False):
            return self
        for layer in self.model.layers:
            if isinstance(layer, LRADDecoderBlock) or "lrad" in layer.__class__.__name__.lower():
                layer.ln1.forward = torch.compile(layer.ln1.forward, mode=mode, dynamic=dynamic)
                layer.attn.chunk_forward = torch.compile(layer.attn.chunk_forward, mode=mode, dynamic=dynamic)
                layer.ln2.forward = torch.compile(layer.ln2.forward, mode=mode, dynamic=dynamic)
                layer.mlp.forward = torch.compile(layer.mlp.forward, mode=mode, dynamic=dynamic)
        self.model.final_norm.forward = torch.compile(self.model.final_norm.forward, mode=mode, dynamic=dynamic)
        self._is_cs_lrad_compiled = True
        return self

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
        num_logits_to_keep: int = 0,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else bool(getattr(self.config, "return_dict", True))

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

        # Support both `logits_to_keep` (new HF GenerationMixin) and `num_logits_to_keep`
        keep = logits_to_keep if logits_to_keep is not None else num_logits_to_keep
        if isinstance(keep, int) and keep > 0 and hidden_states.size(1) > keep:
            hidden_states = hidden_states[:, -keep:, :]
        elif isinstance(keep, torch.Tensor):
            hidden_states = hidden_states[:, keep, :]

        logits = self.lm_head(hidden_states)

        soft_cap = getattr(self.config, "logit_soft_cap", None)
        if soft_cap is not None and soft_cap > 0.0:
            logits = torch.clamp(logits, min=-float(soft_cap), max=float(soft_cap))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, position_ids=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        past_length = _extract_past_length(past_key_values, attention_mask=attention_mask, seq_length=input_ids.size(1))
        if past_key_values is not None:
            if attention_mask is not None:
                position_ids = (attention_mask.sum(dim=-1, keepdim=True) - 1).clamp(min=0)
            else:
                position_ids = torch.full(
                    (input_ids.size(0), 1), past_length, dtype=torch.long, device=input_ids.device
                )

        if position_ids is not None:
            position_ids = position_ids.to(input_ids.device)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
            "num_logits_to_keep": 1,
            "logits_to_keep": 1,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None

        def _reorder_item(item):
            if isinstance(item, DynamicCache):
                item.reorder_cache(beam_idx)
                return item
            elif isinstance(item, torch.Tensor):
                return item.index_select(0, beam_idx)
            elif isinstance(item, (tuple, list)):
                return tuple(_reorder_item(x) for x in item)
            return item

        return tuple(_reorder_item(layer_past) for layer_past in past_key_values)


# Global Model Registration
AutoModel.register(BareTorchConfig, BareTorchModel)
AutoModelForCausalLM.register(BareTorchConfig, BareTorchForCausalLM)
AutoModelForCausalLM.register(CSLRADConfig, CSLRADForCausalLM)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)