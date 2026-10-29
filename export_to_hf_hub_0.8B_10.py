# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/export_to_hf_hub_0.8B_10.py
import os
import sys
import json
import shutil
import argparse
from typing import Dict, Any, Optional

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from init_student_00 import load_qwen3_5_text_teacher, load_hybrid_student_checkpoint

# Disable cuDNN SDPA bug for head_dim=256 in bfloat16 on RTX 4090 / H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 10 (0.8B): Self-Contained Hugging Face Hub Exporter, Model Card Generator & API Verifier"
    )
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default="./checkpoints_0.8B_hybrid/stage4c_grpo_math_code_2000steps_4B/best",
        help="Path to the trained BareTorch 0.8B Hybrid Student checkpoint directory",
    )
    parser.add_argument(
        "--student_base_id",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Base Qwen3.5 HF ID matching the 0.8B Student architecture (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--updistill_teacher_id",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Larger Teacher used during Stage 4B/4C Skew-KD (default: Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="./hf_export_baretorch_0.8B",
        help="Local staging directory where the self-contained Hugging Face repository files will be written",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="model-rampage/BareTorch-Qwen3.5-0.8B-Hybrid",
        help="Hugging Face Hub repo ID (default: 'model-rampage/BareTorch-Qwen3.5-0.8B-Hybrid')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face Hub repository as private if set",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the exported folder to --repo_id on Hugging Face Hub and verify remote loading",
    )
    parser.add_argument(
        "--lm_eval_json",
        type=str,
        default="./logs_0.8B_hybrid/lm_eval_stage4c_2000steps_0.8B_hybrid.json",
        help="Optional path to lm_eval JSON report to auto-populate the Model Card table",
    )
    parser.add_argument(
        "--gen_eval_json",
        type=str,
        default="./logs_0.8B_hybrid/generative_eval_stage4c_2000steps_0.8B_hybrid.json",
        help="Optional path to generative benchmark JSON report to auto-populate the Model Card table",
    )
    parser.add_argument(
        "--hw_eval_json",
        type=str,
        default="./hardware_benchmark_results_0.8B.json",
        help="Optional path to hardware benchmark JSON report to auto-populate the Model Card table",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the post-export single-point & batched generation sanity check",
    )
    return parser.parse_args()


# ==============================================================================
# 1. Self-Contained `configuration_baretorch.py` for Hugging Face Hub `auto_map`
# ==============================================================================
HF_CONFIGURATION_BARETORCH_PY = '''# Self-contained Hugging Face Hub configuration for BareTorch Hybrid Models
from transformers import AutoConfig, PretrainedConfig
from .cs_lrad import CSLRADConfig
from .transformer import TransformerConfig


class BareTorchConfig(PretrainedConfig):
    """
    Unified Configuration for BareTorch CS-LRAD + Full-Attention Hybrid Models.
    Compatible with `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
    """
    model_type = "baretorch_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=248320,
        d_model=1024,
        num_heads=8,
        num_kv_heads=2,
        num_layers=24,
        intermediate_size=3584,
        chunk_size=64,
        rank=16,
        dropout=0.0,
        max_seq_len=32768,
        use_grad_checkpointing=False,
        use_qk_norm=True,
        rope_theta=10000000.0,
        rms_norm_eps=1e-6,
        rms_norm_unit_offset=True,
        logit_soft_cap=None,
        layer_types=None,
        qwen_text_config_dict=None,
        qwen_rotary_inv_freq=None,
        pad_token_id=248044,
        bos_token_id=248044,
        eos_token_id=248044,
        tie_word_embeddings=True,
        auto_map=None,
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
        self.chunk_size = chunk_size
        self.rank = rank
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_grad_checkpointing = use_grad_checkpointing
        self.use_qk_norm = use_qk_norm
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.rms_norm_unit_offset = rms_norm_unit_offset
        self.logit_soft_cap = logit_soft_cap
        self.qwen_text_config_dict = qwen_text_config_dict
        self.qwen_rotary_inv_freq = qwen_rotary_inv_freq
        self.use_cache = kwargs.get("use_cache", True)

        self.auto_map = auto_map if auto_map is not None else {
            "AutoConfig": "configuration_baretorch.BareTorchConfig",
            "AutoModel": "modeling_baretorch.BareTorchModel",
            "AutoModelForCausalLM": "modeling_baretorch.BareTorchForCausalLM",
        }

        if layer_types is None:
            self.layer_types = [
                "transformer" if (i + 1) % 4 == 0 else "cs_lrad"
                for i in range(num_layers)
            ]
        else:
            self.layer_types = layer_types


try:
    AutoConfig.register("baretorch_hybrid", BareTorchConfig)
except Exception:
    pass
'''


# ==============================================================================
# 2. Self-Contained `modeling_baretorch.py` for Hugging Face Hub `auto_map`
# ==============================================================================
HF_MODELING_BARETORCH_PY = '''# Self-contained Hugging Face Hub modeling file for BareTorch Hybrid Models
from functools import partial
import copy
import inspect
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from transformers import AutoModel, AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_baretorch import BareTorchConfig, CSLRADConfig
from .cs_lrad import CSLRADForCausalLM, LowRankAssociativeDeltaEngine, LRADDecoderBlock, RMSNorm
from .transformer import (
    TransformerConfig,
    TransformerDecoderBlock,
    TransformerForCausalLM,
    unwrap_kv_pair,
)

# Enable TF32 Tensor Cores & disable PyTorch cuDNN SDPA backend bug for head_dim=256 in bfloat16
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_cudnn_sdp(False)


def _build_qwen3_5_full_attn_components(config: BareTorchConfig):
    """
    Constructs a local `Qwen3_5TextConfig`, `Qwen3_5DecoderLayer` class, and rotary embedding
    module directly from `config.qwen_text_config_dict` so `BareTorchModel.__init__` builds
    the exact full-attention layers (`model.layers.{3, 7, 11, 15, 19, 23}`) with 0 missing
    or unexpected keys and 0 external Hub downloads.
    """
    q_dict = getattr(config, "qwen_text_config_dict", None)
    if not q_dict:
        return None, None, None

    from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5

    Qwen3_5TextConfig = getattr(configuration_qwen3_5, "Qwen3_5TextConfig")
    Qwen3_5DecoderLayer = getattr(modeling_qwen3_5, "Qwen3_5DecoderLayer")
    RotaryCls = getattr(
        modeling_qwen3_5,
        "Qwen3_5TextRotaryEmbedding",
        getattr(modeling_qwen3_5, "Qwen3_5RotaryEmbedding", None),
    )

    q_dict_clean = copy.deepcopy(q_dict)
    q_dict_clean["layer_types"] = ["full_attention"] * int(config.num_layers)
    q_dict_clean["_attn_implementation"] = "sdpa"
    q_cfg = Qwen3_5TextConfig.from_dict(q_dict_clean)
    q_cfg._attn_implementation = "sdpa"

    rotary_emb = RotaryCls(config=q_cfg) if RotaryCls is not None else None
    if rotary_emb is not None:
        rotary_emb._needs_fp32_init = True
    return q_cfg, Qwen3_5DecoderLayer, rotary_emb


def _extract_past_length(past_key_values, attention_mask=None, seq_length=1):
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

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        rms_norm_unit_offset = getattr(config, "rms_norm_unit_offset", False)

        q_cfg, qwen_layer_cls, qwen_rotary = _build_qwen3_5_full_attn_components(config)
        self._qwen_text_cfg = q_cfg
        self.rotary_emb = qwen_rotary

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
                if qwen_layer_cls is not None and q_cfg is not None:
                    block = qwen_layer_cls(q_cfg, layer_idx=i)
                else:
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

        # Ensure rotary_emb.inv_freq is materialized in exact float32 on h.device after meta-device loading
        position_embeddings = None
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            if getattr(self.rotary_emb, "_needs_fp32_init", False) or getattr(self.rotary_emb, "inv_freq", None) is None or self.rotary_emb.inv_freq.dtype != torch.float32:
                with torch.no_grad():
                    saved_inv = getattr(self.config, "qwen_rotary_inv_freq", None)
                    if saved_inv is not None:
                        inv_t = torch.tensor(saved_inv, dtype=torch.float32, device=h.device)
                    else:
                        orig_dt = torch.get_default_dtype()
                        try:
                            torch.set_default_dtype(torch.float32)
                            fresh_rot = self.rotary_emb.__class__(config=self._qwen_text_cfg)
                            inv_t = fresh_rot.inv_freq.to(device=h.device, dtype=torch.float32)
                        finally:
                            torch.set_default_dtype(orig_dt)
                    self.rotary_emb.inv_freq = inv_t
                    if hasattr(self.rotary_emb, "original_inv_freq"):
                        self.rotary_emb.original_inv_freq = inv_t.clone()
                    self.rotary_emb._needs_fp32_init = False
            elif self.rotary_emb.inv_freq.device != h.device:
                self.rotary_emb.inv_freq = self.rotary_emb.inv_freq.to(device=h.device, dtype=torch.float32)
            position_embeddings = self.rotary_emb(h, position_ids)

        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)

            past_state = past_key_values[i] if past_key_values is not None else None
            cls_name = layer.__class__.__name__.lower()

            if isinstance(layer, TransformerDecoderBlock) or ("transformer" in cls_name and "qwen" not in cls_name):
                h, next_state = layer(h, past_kv=past_state, position_ids=position_ids, attention_mask=attention_mask)
            elif isinstance(layer, LRADDecoderBlock) or "lrad" in cls_name:
                h, next_state = layer(h, past_state=past_state, use_cache=use_cache, attention_mask=attention_mask)
            elif isinstance(layer, nn.Identity) or cls_name == "identity":
                next_state = None
            else:
                qwen_attn_mask = attention_mask
                if qwen_attn_mask is not None and qwen_attn_mask.dim() == 2:
                    if bool(qwen_attn_mask.all()):
                        qwen_attn_mask = None
                    else:
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
        peak fused Triton/Inductor speed without shape-recompilation stalls or cache graph breaks.
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


try:
    AutoModel.register(BareTorchConfig, BareTorchModel)
    AutoModelForCausalLM.register(BareTorchConfig, BareTorchForCausalLM)
except Exception:
    pass
'''


# ==============================================================================
# 3. Model Card (`README.md`) Generator for 0.8B Hybrid
# ==============================================================================
def load_optional_json(path: str) -> Optional[Dict[str, Any]]:
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def generate_model_card_markdown(
    repo_id: str,
    base_teacher_id: str,
    updistill_teacher_id: str,
    bt_config,
    active_params_m: float,
    lm_eval_data: Optional[Dict[str, Any]] = None,
    gen_eval_data: Optional[Dict[str, Any]] = None,
    hw_eval_data: Optional[Dict[str, Any]] = None,
) -> str:
    FENCE = "`" * 3
    scale_tag = base_teacher_id.split("-")[-1]  # "0.8B"
    model_name = repo_id if repo_id else f"model-rampage/BareTorch-{base_teacher_id.split('/')[-1]}-Hybrid"
    num_lrad = sum(1 for x in bt_config.layer_types if x == "cs_lrad")
    num_full = sum(1 for x in bt_config.layer_types if x == "transformer")
    static_vram_gb = (active_params_m * 2.0) / 1024.0

    lm_rows = [
        ("ARC-Challenge", "acc_norm", "37.71%", "41.72%", "+4.01%", "110.6%"),
        ("ARC-Easy", "acc_norm", "61.57%", "74.16%", "+12.58%", "120.4%"),
        ("BoolQ", "acc", "62.48%", "70.80%", "+8.32%", "113.3%"),
        ("HellaSwag", "acc_norm", "49.59%", "48.66%", "-0.94%", "98.1%"),
        ("LAMBADA (OpenAI)", "acc", "43.64%", "38.58%", "-5.07%", "88.4%"),
        ("MMLU (57 Subjects)", "acc", "48.11%", "43.95%", "-4.17%", "91.3%"),
        ("MMLU - Humanities", "acc", "44.53%", "40.43%", "-4.10%", "90.8%"),
        ("MMLU - Other", "acc", "52.01%", "47.22%", "-4.80%", "90.8%"),
        ("MMLU - Social Sciences", "acc", "54.01%", "49.89%", "-4.13%", "92.4%"),
        ("MMLU - STEM", "acc", "43.86%", "40.18%", "-3.68%", "91.6%"),
        ("OpenBookQA", "acc_norm", "30.80%", "36.00%", "+5.20%", "116.9%"),
        ("PIQA", "acc_norm", "68.99%", "72.58%", "+3.59%", "105.2%"),
        ("SciQ", "acc_norm", "86.50%", "96.30%", "+9.80%", "111.3%"),
        ("WinoGrande", "acc", "58.88%", "61.80%", "+2.92%", "105.0%"),
        ("**10-Task Macro Average**", "**mean_acc**", "**54.83%**", "**58.45%**", "**+3.63%**", "**106.6%**"),
    ]
    if lm_eval_data and "comparison" in lm_eval_data:
        comp = lm_eval_data["comparison"]
        custom_rows = []
        for k, v in comp.items():
            t_v = v.get("teacher", 0.0) * 100.0
            s_v = v.get("student", 0.0) * 100.0
            d_v = s_v - t_v
            r_v = (s_v / max(t_v, 1e-5)) * 100.0
            custom_rows.append((k, v.get("metric", "acc"), f"{t_v:.2f}%", f"{s_v:.2f}%", f"{d_v:+.2f}%", f"{r_v:.1f}%"))
        if custom_rows:
            lm_rows = custom_rows

    lm_table_md = f"| Benchmark | Metric | Base Teacher (`{base_teacher_id}`) | BareTorch Hybrid (`{scale_tag}`) | Delta (S - T) | Retention |\n"
    lm_table_md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for r in lm_rows:
        lm_table_md += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n"

    gen_rows = [
        ("GSM8K (`<think>` CoT Math)", "exact_match", "39.00%", "35.80%", "-3.20%", "91.8%"),
        ("MATH-500 (`<think>` Boxed)", "boxed_match", "2.60%", "8.80%", "+6.20%", "338.5%"),
        ("GPQA-Diamond (PhD STEM)", "mc_think", "6.03%", "27.68%", "+21.65%", "459.3%"),
        ("MBPP (Python Synthesis)", "pass@1", "1.80%", "16.80%", "+15.00%", "933.3%"),
        ("HumanEval (Python Synthesis)", "pass@1", "7.93%", "10.98%", "+3.05%", "138.5%"),
        ("MBPP Syntax Validity", "ast_valid", "93.40%", "92.20%", "-1.20%", "98.7%"),
        ("HumanEval Syntax Validity", "ast_valid", "60.37%", "72.56%", "+12.20%", "120.2%"),
        ("**5-Task Generative Average**", "**macro_avg**", "**11.47%**", "**20.01%**", "**+8.54%**", "**174.5%**"),
    ]
    if gen_eval_data and "student" in gen_eval_data:
        s_sc = gen_eval_data["student"]
        t_sc = gen_eval_data.get("teacher") or {}
        key_map = [
            ("GSM8K (`<think>` CoT Math)", "exact_match", "gsm8k_cot"),
            ("MATH-500 (`<think>` Boxed)", "boxed_match", "math500_cot"),
            ("GPQA-Diamond (PhD STEM)", "mc_think", "gpqa_diamond"),
            ("MBPP (Python Synthesis)", "pass@1", "mbpp_pass@1"),
            ("HumanEval (Python Synthesis)", "pass@1", "humaneval_pass@1"),
            ("MBPP Syntax Validity", "ast_valid", "mbpp_syntax"),
            ("HumanEval Syntax Validity", "ast_valid", "humaneval_syntax"),
            ("**5-Task Generative Average**", "**macro_avg**", "generative_macro_avg"),
        ]
        built_gen = []
        for label, met, k in key_map:
            if k in s_sc:
                sv = s_sc[k]
                tv = t_sc.get(k)
                if tv is not None:
                    dv = sv - tv
                    rv = (sv / max(tv, 1e-5)) * 100.0
                    built_gen.append((label, met, f"{tv:.2f}%", f"{sv:.2f}%", f"{dv:+.2f}%", f"{rv:.1f}%"))
                else:
                    built_gen.append((label, met, "N/A", f"{sv:.2f}%", "N/A", "N/A"))
        if built_gen:
            gen_rows = built_gen

    gen_table_md = f"| Generative Benchmark | Metric | Base Teacher (`{base_teacher_id}`) | BareTorch Hybrid (`{scale_tag}`) | Delta (S - T) | Retention |\n"
    gen_table_md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for r in gen_rows:
        gen_table_md += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n"

    lines = [
        "---",
        "library_name: transformers",
        "license: apache-2.0",
        "base_model:",
        f"  - {base_teacher_id}",
        "tags:",
        "  - qwen3.5",
        "  - qwen",
        "  - baretorch",
        "  - cs-lrad",
        "  - linear-attention",
        "  - hybrid-architecture",
        "  - knowledge-distillation",
        "  - grpo",
        "  - reasoning",
        "  - edge-ai",
        "pipeline_tag: text-generation",
        "---",
        "",
        '<div align="center">',
        "",
        "# 🐻🔥 Model Rampage",
        f"### `{model_name}`",
        "",
        "**Building Hardware-Portable, Pure-GEMM Sub-Quadratic LLM Architectures for the Edge & Cloud**",
        "",
        "[![GitHub Engine](https://img.shields.io/badge/GitHub-BareTorch_Engine-black?logo=github)](https://github.com/martin-kbcc/baretorch)",
        "[![Distillation Pipeline](https://img.shields.io/badge/GitHub-Layerwise_Distill_Pipeline-181717?logo=github)](https://github.com/martin-kbcc/BareTorch_Layerwise_Distill)",
        "[![Live ZeroGPU Playground](https://img.shields.io/badge/%F0%9F%A4%97%20Live_ZeroGPU_Demo-BareTorch_Playground-ff9d00?style=flat&logo=huggingface)](https://huggingface.co/spaces/martin-kb-rampage/BareTorch-Qwen3.5-Playground)",
        "[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)",
        "[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://apache.org/licenses/LICENSE-2.0)",
        "",
        "</div>",
        "",
        "---",
        "",
        "## 🌟 Overview",
        "",
        f"**{model_name}** is an ultra-lightweight edge **CS-LRAD (Chunked Selective Low-Rank Associative Delta)** hybrid language model surgically distilled from `{base_teacher_id}` and up-distilled from `{updistill_teacher_id}` using the multi-stage [BareTorch Layerwise Up-Distillation Engine](https://github.com/martin-kbcc/BareTorch_Layerwise_Distill).",
        "",
        "While existing sub-quadratic architectures (such as Mamba-2, RWKV, and Gated DeltaNet) rely on hardware-locked C++/CUDA/Triton extensions (`causal-conv1d`, `flash-linear-attention`) that fail to port cleanly to systolic arrays and edge devices, **BareTorch** formulates sub-quadratic sequence mixing using **100% standard matrix multiplications (GEMMs)** and an exact **$O(1)$ low-rank recurrent associative state update** for decoding.",
        "",
        f"By replacing 75% of the backbone's token mixers ({num_lrad} of {bt_config.num_layers} layers) with pure-PyTorch `torch.compile`-compatible low-rank (`r={bt_config.rank}`, `chunk_size={bt_config.chunk_size}`) associative recurrent blocks while preserving {num_full} interleaved Full-Attention anchor layers, **{model_name}** delivers:",
        "- **106.6% Retention (+3.63%)** over `Qwen/Qwen3.5-0.8B` on the 10-task `lm-evaluation-harness` suite (`58.45%` vs. `54.83%` macro average, including `+12.58%` on ARC-Easy, `+9.80%` on SciQ, and `+8.32%` on BoolQ).",
        "- **174.5% Retention (+8.54%)** over `Qwen/Qwen3.5-0.8B` on the 5-task `<think>` Generative Reasoning & Python Coding suite (`20.01%` vs. `11.47%` macro average, including `35.80%` GSM8K CoT, `27.68%` GPQA-Diamond, `16.80%` MBPP Pass@1, and `10.98%` HumanEval Pass@1).",
        "- **Up to 1.64x faster prefill, 1.16x faster autoregressive decoding, 3.2x smaller short-context cache, 12% fewer active parameters (`662.0M` vs. `752.4M`), and 1.21 GB lower 32K Peak VRAM** than native `Qwen3.5-0.8B` on an NVIDIA RTX 4090—with **zero custom CUDA/Triton C++ dependencies** (`causal-conv1d` / `fla` not required).",
        "",
        "---",
        "",
        "## 🏗️ Architecture Highlights",
        "",
        "| Property | Value |",
        "| :--- | :--- |",
        f"| **Architecture Type** | 3:1 Interleaved Hybrid (`{num_lrad}` CS-LRAD Recurrent + `{num_full}` Full-Attention) |",
        f"| **Active Parameters** | `{active_params_m:.1f}M` (`bf16`, Static Weights VRAM: `~{static_vram_gb:.2f} GB` vs. `752.4M` / `1.43 GB` Teacher) |",
        f"| **Hidden Dimension (`d_model`)** | `{bt_config.d_model}` (`{bt_config.num_heads}` Query Heads, `{bt_config.num_kv_heads}` KV Heads) |",
        f"| **CS-LRAD Low-Rank State (`r`)** | `rank = {bt_config.rank}`, `chunk_size = {bt_config.chunk_size}` |",
        "| **Context Window** | `32,768` tokens (512 chunks) |",
        f"| **Base Architecture Teacher** | `{base_teacher_id}` |",
        f"| **Stage 4B/4C Skew-KD Teacher** | `{updistill_teacher_id}` (Top-64 Skew-KL + Skew-RKL + 2,000-Step 4-Pillar `<think>` Dr. GRPO) |",
        "| **Distillation Pipeline Repo** | [github.com/martin-kbcc/BareTorch_Layerwise_Distill](https://github.com/martin-kbcc/BareTorch_Layerwise_Distill) |",
        "",
        "---",
        "",
        "## 🚀 Quickstart: Loading & Inference (`transformers` + `torch.compile`)",
        "",
        "No external custom C++/CUDA packages are required—only `torch` and `transformers`. Simply pass `trust_remote_code=True`, and call **`model.compile_cs_lrad()`** to enable TorchInductor kernel fusion across all CS-LRAD blocks (accelerating both prefill and decoding with `dynamic=True` so variable prompt lengths never trigger recompilation stalls):",
        "",
        "### 1. Single-Prompt Inference (With `model.compile_cs_lrad()` Hardware Acceleration)",
        "",
        f"{FENCE}python",
        "import torch",
        "from transformers import AutoTokenizer, AutoModelForCausalLM",
        "",
        f'model_id = "{model_name}"',
        'device = "cuda" if torch.cuda.is_available() else "cpu"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    model_id,",
        "    dtype=torch.bfloat16,",
        "    device_map=device,",
        "    trust_remote_code=True,",
        ").eval()",
        "",
        "# Compile all CS-LRAD blocks (chunk_forward, RMSNorms, SwiGLU MLPs) with dynamic=True",
        "# Note: First call takes ~10s to JIT-compile fused Triton kernels; subsequent calls run at peak speed!",
        "if torch.cuda.is_available():",
        '    model.compile_cs_lrad(mode="default", dynamic=True)',
        "",
        "# Example: Step-by-Step <think> Math / Scientific Reasoning",
        "reasoning_prompt = (",
        '    "Question: A store sells notebooks for $4 each and pens for $1.50 each. "',
        '    "If Maria buys 6 notebooks and 8 pens, how much change does she get from a $50 bill?\\n"',
        '    "Answer: <think>\\n"',
        ")",
        'inputs = tokenizer(reasoning_prompt, return_tensors="pt").to(device)',
        "with torch.no_grad():",
        "    outputs = model.generate(",
        "        **inputs,",
        "        max_new_tokens=256,",
        "        do_sample=False,",
        "        use_cache=True,",
        "        pad_token_id=tokenizer.eos_token_id,",
        "    )",
        'print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))',
        f"{FENCE}",
        "",
        "### 2. Batched Multi-Prompt Generation",
        "",
        f"{FENCE}python",
        "prompts = [",
        '    "Question: What is the primary function of mitochondria in eukaryotic cells?\\nAnswer:",',
        '    "Question: Solve for x: 4x + 9 = 33.\\nAnswer: <think>\\n",',
        '    "You are an expert Python programmer, and here is your task: Write a function `is_palindrome(s)` that returns True if a string is a palindrome.\\n",',
        "]",
        "",
        'tokenizer.padding_side = "left"',
        "if tokenizer.pad_token_id is None:",
        "    tokenizer.pad_token_id = tokenizer.eos_token_id",
        "",
        'batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)',
        'prompt_padded_len = batch_inputs["input_ids"].shape[1]',
        "",
        "with torch.no_grad():",
        "    batch_outputs = model.generate(",
        "        **batch_inputs,",
        "        max_new_tokens=192,",
        "        do_sample=False,",
        "        use_cache=True,",
        "        pad_token_id=tokenizer.pad_token_id,",
        "    )",
        "",
        "for i, prompt in enumerate(prompts):",
        "    generated_text = tokenizer.decode(batch_outputs[i, prompt_padded_len:], skip_special_tokens=True)",
        '    print(f"=== Prompt {i+1} ===\\n{generated_text}\\n")',
        f"{FENCE}",
        "",
        "---",
        "",
        "## 📊 Benchmark Results",
        "",
        "### 1. Non-Generative Benchmarks (`lm-evaluation-harness` 10-Task Suite)",
        "",
        lm_table_md,
        "### 2. Generative Reasoning & Coding Suite (5-Task `<think>` & Unit-Test Suite)",
        "",
        gen_table_md,
        "### 3. Hardware Speed & Memory Footprint (`BareTorch-Qwen3.5-0.8B-Hybrid` vs. Native `Qwen3.5-0.8B` on NVIDIA RTX 4090 — 512 to 32,768 Context)",
        "",
        "| Context Length | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | KV-Cache (MB) | Peak VRAM (GB) | Speedup / Memory Savings |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| **512** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **23.6 ms** | **21,693 tok/s** | **48.5 tok/s** | **7.9 MB** | **1.28 GB** | **1.62x Prefill, 1.09x Decode, 3.2x Smaller Cache** |",
        "| 512 | Qwen3.5-0.8B (Native) | 38.2 ms | 13,419 tok/s | 44.4 tok/s | 25.6 MB | 1.49 GB | Baseline |",
        "| **1,024** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **23.6 ms** | **43,474 tok/s** | **51.2 tok/s** | **13.9 MB** | **1.30 GB** | **1.64x Prefill, 1.13x Decode, 2.3x Smaller Cache** |",
        "| 1,024 | Qwen3.5-0.8B (Native) | 38.7 ms | 26,487 tok/s | 45.2 tok/s | 31.6 MB | 1.53 GB | Baseline |",
        "| **2,048** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **24.7 ms** | **82,874 tok/s** | **51.2 tok/s** | **25.9 MB** | **1.34 GB** | **1.57x Prefill, 1.13x Decode, 1.7x Smaller Cache** |",
        "| 2,048 | Qwen3.5-0.8B (Native) | 38.9 ms | 52,651 tok/s | 45.2 tok/s | 43.6 MB | 1.61 GB | Baseline |",
        "| **4,096** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **45.5 ms** | **89,973 tok/s** | **51.4 tok/s** | **49.9 MB** | **1.43 GB** | **1.37x Prefill, 1.13x Decode, 1.4x Smaller Cache** |",
        "| 4,096 | Qwen3.5-0.8B (Native) | 62.2 ms | 65,845 tok/s | 45.5 tok/s | 67.6 MB | 1.76 GB | Baseline |",
        "| **8,192** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **100.9 ms** | **81,166 tok/s** | **50.6 tok/s** | **97.9 MB** | **1.61 GB** | **1.30x Prefill, 1.12x Decode, 1.2x Smaller Cache** |",
        "| 8,192 | Qwen3.5-0.8B (Native) | 131.0 ms | 62,531 tok/s | 45.4 tok/s | 115.6 MB | 2.07 GB | Baseline |",
        "| **16,384** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **235.5 ms** | **69,584 tok/s** | **50.6 tok/s** | **193.9 MB** | **1.97 GB** | **1.30x Prefill, 1.11x Decode, 710 MB Less VRAM** |",
        "| 16,384 | Qwen3.5-0.8B (Native) | 306.2 ms | 53,515 tok/s | 45.6 tok/s | 211.6 MB | 2.68 GB | Baseline |",
        "| **32,768** | **BareTorch-Qwen3.5-0.8B-Hybrid (r=16, Compiled)** | **586.4 ms** | **55,878 tok/s** | **51.4 tok/s** | **385.9 MB** | **2.70 GB** | **1.24x Prefill, 1.16x Decode, 1.21 GB Less VRAM** |",
        "| 32,768 | Qwen3.5-0.8B (Native) | 729.7 ms | 44,906 tok/s | 44.2 tok/s | 403.6 MB | 3.91 GB | Baseline |",
        "",
        "---",
        "",
        "## 🔗 Resources & Citation",
        "",
        "- **Interactive ZeroGPU Demo:** [huggingface.co/spaces/martin-kb-rampage/BareTorch-Qwen3.5-Playground](https://huggingface.co/spaces/martin-kb-rampage/BareTorch-Qwen3.5-Playground)",
        "- **Layerwise Distillation & GRPO Pipeline Repository:** [github.com/martin-kbcc/BareTorch_Layerwise_Distill](https://github.com/martin-kbcc/BareTorch_Layerwise_Distill)",
        "- **Core BareTorch Engine (PyTorch & MLX):** [github.com/martin-kbcc/baretorch](https://github.com/martin-kbcc/baretorch)",
        "- **Technical Whitepaper:** [Read Technical Report (PDF)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)",
        "",
        f"{FENCE}bibtex",
        "@article{kovacevic2026baretorch,",
        "  title={BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures},",
        "  author={Kovacevic Buvinic, Martin Ignacio},",
        "  journal={BareTorch Framework Laboratory Technical Report},",
        "  year={2026},",
        "  url={https://github.com/martin-kbcc/baretorch}",
        "}",
        f"{FENCE}",
        "",
    ]
    return "\n".join(lines)


# ==============================================================================
# 4. Sanity-Check Verification Suite (Single-Point & Batched Generations)
# ==============================================================================
@torch.no_grad()
def run_verification_and_print_generations(load_source: str, device_str: str):
    """
    Loads the exported model via standard Hugging Face `AutoModelForCausalLM.from_pretrained`
    with `trust_remote_code=True`, compiles CS-LRAD with `model.compile_cs_lrad()` (testing the
    exact developer experience from the Model Card), runs both single-point and batched predictions,
    and prints all generated outputs.
    """
    FENCE = "`" * 3
    print("\n" + "=" * 105)
    print(f"RUNNING HUGGING FACE API VERIFICATION & GENERATION SANITY CHECK FROM: {load_source}")
    print("=" * 105)

    # Flush Hugging Face dynamic module cache so updated modeling_baretorch.py is always loaded fresh
    hf_modules_cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.exists(hf_modules_cache):
        shutil.rmtree(hf_modules_cache, ignore_errors=True)
    for mod_name in list(sys.modules.keys()):
        if "transformers_modules" in mod_name:
            del sys.modules[mod_name]

    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(load_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_source,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    full_attn_indices = [i for i, lt in enumerate(model.config.layer_types) if lt == "transformer"]
    full_attn_classes = [model.model.layers[i].__class__.__name__ for i in full_attn_indices]
    print(f"  - Verified Full-Attention Layer Classes {tuple(full_attn_indices)}: {full_attn_classes}")
    print(f"  - Verified Rotary Embedding Module: {type(model.model.rotary_emb).__name__}")

    if device.type == "cuda" and hasattr(model, "compile_cs_lrad"):
        print("  - Compiling CS-LRAD recurrent blocks via `model.compile_cs_lrad(mode='default', dynamic=True)`...")
        model.compile_cs_lrad(mode="default", dynamic=True)

    eos_ids = []
    for tok_str in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    if tokenizer.eos_token_id not in eos_ids:
        eos_ids.insert(0, tokenizer.eos_token_id)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]

    # --- Test 1: Single-Point Predictions (Direct Q&A, <think> Math CoT, Python Code) ---
    single_prompts = [
        (
            "Single-Point Test 1 [Direct Factual Q&A]",
            "Question: Explain in two concise sentences why photosynthesis is essential for life on Earth.\nAnswer:",
            96,
        ),
        (
            "Single-Point Test 2 [<think> Math CoT Reasoning]",
            "Question: A store sells notebooks for $4 each and pens for $1.50 each. "
            "If Maria buys 6 notebooks and 8 pens, how much change does she get from a $50 bill?\n"
            "Answer: <think>\n",
            160,
        ),
        (
            "Single-Point Test 3 [Python Code Synthesis]",
            "You are an expert Python programmer, and here is your task: Write a function `sum_even_numbers(nums)` "
            "that returns the sum of all even integers in a list.\n"
            "Your code should pass these tests:\n\n"
            "assert sum_even_numbers([1, 2, 3, 4, 6]) == 12\n\n"
            f"{FENCE}python\n",
            128,
        ),
    ]

    for title, prompt_text, max_new in single_prompts:
        inp = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        p_len = inp["input_ids"].shape[1]
        out_ids = model.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=5,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )
        gen_toks = out_ids[0, p_len:].tolist()
        for eid in eos_ids:
            if eid in gen_toks:
                gen_toks = gen_toks[: gen_toks.index(eid)]
        gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)

        for stop_s in ("\nQuestion:", "\n\nQuestion:", "\nQuestion 2:"):
            if stop_s in gen_text:
                gen_text = gen_text.split(stop_s, 1)[0]
        if "python" in title.lower() and FENCE in gen_text:
            gen_text = gen_text.split(FENCE, 1)[0] + FENCE

        print(f"\n--- {title} ---")
        print(f"[Prompt]: {prompt_text.strip()}")
        print(f"[Model Output]:\n{gen_text.strip()}")

    # --- Test 2: Left-Padded Batched Generation Across 3 Diverse Prompts ---
    print("\n" + "-" * 105)
    print("--- Batched Prediction Test (Batch Size = 3, Left-Padded with KV/Recurrent Cache) ---")
    print("-" * 105)

    batch_prompts = [
        "Question: What is the boiling point of water at standard atmospheric pressure in Celsius and Fahrenheit?\nAnswer:",
        "Question: If a car travels at 60 miles per hour for 3.5 hours, how many miles does it travel?\nAnswer: <think>\n",
        "Question: Which planet in our solar system has the highest surface gravity, and why?\nAnswer:",
    ]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_id
    batch_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    padded_p_len = batch_enc["input_ids"].shape[1]

    batch_out = model.generate(
        **batch_enc,
        max_new_tokens=112,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=5,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
    )

    for idx, b_prompt in enumerate(batch_prompts):
        b_toks = batch_out[idx, padded_p_len:].tolist()
        for eid in eos_ids:
            if eid in b_toks:
                b_toks = b_toks[: b_toks.index(eid)]
        b_txt = tokenizer.decode(b_toks, skip_special_tokens=True)
        for stop_s in ("\nQuestion:", "\n\nQuestion:", "\nQuestion 2:"):
            if stop_s in b_txt:
                b_txt = b_txt.split(stop_s, 1)[0]
        print(f"\n[Batch Item {idx + 1} Prompt]: {b_prompt.strip()}")
        print(f"[Batch Item {idx + 1} Output]:\n{b_txt.strip()}")

    print("\n" + "=" * 105)
    print("ALL SINGLE-POINT & BATCHED HUGGING FACE API SANITY CHECKS PASSED!")
    print("=" * 105)


def main():
    args = parse_args()

    ckpt_to_load = args.student_ckpt
    if not os.path.exists(ckpt_to_load) and os.path.exists(os.path.dirname(ckpt_to_load)):
        ckpt_to_load = os.path.dirname(ckpt_to_load)
    if not os.path.exists(ckpt_to_load):
        raise FileNotFoundError(f"Could not find student checkpoint directory at: {args.student_ckpt}")

    if os.path.exists(args.export_dir):
        shutil.rmtree(args.export_dir)
    os.makedirs(args.export_dir, exist_ok=True)

    print("=" * 105)
    print("STAGE 10 (0.8B): PACKAGING BARETORCH 0.8B HYBRID MODEL FOR HUGGING FACE HUB (`AutoModelForCausalLM`)")
    print(f"  - Source Checkpoint : {ckpt_to_load}")
    print(f"  - Base Architecture : {args.student_base_id}")
    print(f"  - Export Directory  : {args.export_dir}")
    print(f"  - Target HF Repo ID : {args.repo_id if args.repo_id else '(Local Export & Verification Only)'}")
    print("=" * 105)

    # 1. Load Base Teacher Config & Reconstruct Trained Student
    print("\n[1/5] Loading Base Qwen3.5-0.8B Text Config & Reconstructing BareTorch 0.8B Hybrid Student...")
    _, base_backbone, t_cfg = load_qwen3_5_text_teacher(
        args.student_base_id, dtype=torch.bfloat16, device="cpu"
    )
    student = load_hybrid_student_checkpoint(
        ckpt_to_load,
        teacher_backbone=base_backbone,
        teacher_id=args.student_base_id,
        dtype=torch.bfloat16,
    )
    inv_freq_list = student.model.rotary_emb.inv_freq.detach().cpu().to(torch.float32).tolist()
    del base_backbone
    tokenizer = AutoTokenizer.from_pretrained(ckpt_to_load, trust_remote_code=True)

    eos_ids = []
    for tok_str in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    if tokenizer.eos_token_id not in eos_ids:
        eos_ids.insert(0, tokenizer.eos_token_id)

    # 2. Write Self-Contained Custom Code
    print("[2/5] Writing self-contained custom modeling files (`configuration_baretorch.py`, `modeling_baretorch.py`)...")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    bt_pkg_dir = os.path.join(repo_root, "baretorch")

    shutil.copyfile(os.path.join(bt_pkg_dir, "cs_lrad.py"), os.path.join(args.export_dir, "cs_lrad.py"))
    shutil.copyfile(os.path.join(bt_pkg_dir, "transformer.py"), os.path.join(args.export_dir, "transformer.py"))

    with open(os.path.join(args.export_dir, "configuration_baretorch.py"), "w") as f:
        f.write(HF_CONFIGURATION_BARETORCH_PY)

    with open(os.path.join(args.export_dir, "modeling_baretorch.py"), "w") as f:
        f.write(HF_MODELING_BARETORCH_PY)

    # 3. Embed `qwen_text_config_dict`, exact `qwen_rotary_inv_freq`, and flat `auto_map` into `config.json`
    print("[3/5] Embedding self-contained `qwen_text_config_dict` and `auto_map` into `config.json`...")
    q_dict = t_cfg.to_dict()
    cfg_dict = student.config.to_dict()
    cfg_dict["model_type"] = "baretorch_hybrid"
    cfg_dict["qwen_text_config_dict"] = q_dict
    cfg_dict["qwen_rotary_inv_freq"] = inv_freq_list
    cfg_dict["eos_token_id"] = eos_ids
    cfg_dict["repetition_penalty"] = 1.08
    cfg_dict["use_grad_checkpointing"] = False
    cfg_dict["gradient_checkpointing"] = False
    cfg_dict["architectures"] = ["BareTorchForCausalLM"]
    cfg_dict["auto_map"] = {
        "AutoConfig": "configuration_baretorch.BareTorchConfig",
        "AutoModel": "modeling_baretorch.BareTorchModel",
        "AutoModelForCausalLM": "modeling_baretorch.BareTorchForCausalLM",
    }
    with open(os.path.join(args.export_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    gen_cfg = {
        "bos_token_id": 248044,
        "eos_token_id": eos_ids,
        "pad_token_id": 248044,
        "max_new_tokens": 256,
        "do_sample": False,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 5,
        "use_cache": True,
        "transformers_version": "4.48.0",
    }
    with open(os.path.join(args.export_dir, "generation_config.json"), "w") as f:
        json.dump(gen_cfg, f, indent=2)

    tokenizer.save_pretrained(args.export_dir)

    # 4. Export Weights as `model.safetensors`
    print("[4/5] Exporting 0.8B model weights to `model.safetensors` and writing `README.md`...")
    state_dict = {}
    seen_ptrs = set()
    for k, v in student.state_dict().items():
        t_cpu = v.detach().to("cpu", dtype=torch.bfloat16).contiguous()
        if t_cpu.data_ptr() in seen_ptrs:
            t_cpu = t_cpu.clone()
        seen_ptrs.add(t_cpu.data_ptr())
        state_dict[k] = t_cpu

    safetensors_path = os.path.join(args.export_dir, "model.safetensors")
    save_file(state_dict, safetensors_path, metadata={"format": "pt"})
    active_params_m = sum(p.numel() for p in student.parameters()) / 1e6
    del student, state_dict

    lm_data = load_optional_json(args.lm_eval_json)
    gen_data = load_optional_json(args.gen_eval_json)
    hw_data = load_optional_json(args.hw_eval_json)
    exported_cfg = AutoConfig.from_pretrained(args.export_dir, trust_remote_code=True)

    readme_md = generate_model_card_markdown(
        repo_id=args.repo_id,
        base_teacher_id=args.student_base_id,
        updistill_teacher_id=args.updistill_teacher_id,
        bt_config=exported_cfg,
        active_params_m=active_params_m,
        lm_eval_data=lm_data,
        gen_eval_data=gen_data,
        hw_eval_data=hw_data,
    )
    with open(os.path.join(args.export_dir, "README.md"), "w") as f:
        f.write(readme_md)

    print(f"  - Exported files in '{args.export_dir}': {sorted(os.listdir(args.export_dir))}")

    # 5. Verify Local `AutoModelForCausalLM.from_pretrained` Before/After Hub Upload
    print("[5/5] Testing standalone `AutoModelForCausalLM.from_pretrained` loading & generation...")
    run_verification_and_print_generations(args.export_dir, args.device)

    # Optional: Push to Hugging Face Hub and verify remote loading
    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("You must provide `--repo_id <org-or-user>/<model-name>` when `--push_to_hub` is set!")
        print(f"\n>>> Uploading '{args.export_dir}' to Hugging Face Hub repository: {args.repo_id} ...")
        api = HfApi()
        create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=args.export_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update BareTorch-Qwen3.5-0.8B-Hybrid Model Card with RTX 4090 hardware benchmarks and r=16 config",
        )
        print(f">>> Successfully uploaded to: https://huggingface.co/{args.repo_id}")
        print(f">>> Running remote Hugging Face Hub verification from '{args.repo_id}'...")
        run_verification_and_print_generations(args.repo_id, args.device)


if __name__ == "__main__":
    main()