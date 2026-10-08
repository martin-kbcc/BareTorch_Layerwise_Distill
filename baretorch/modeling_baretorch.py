# /home/martinkb/Desktop/BareTorch_Layerwise_Distill/baretorch/modeling_baretorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from .configuration_baretorch import BareTorchConfig, CSLRADConfig
from .cs_lrad import LRADDecoderBlock, CSLRADForCausalLM, RMSNorm


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

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None, *args, **kwargs):
        """
        Flexible gradient checkpointing setter accepting arbitrary Hugging Face kwargs
        (e.g., every_n_layers, value) across transformers version updates.
        """
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
        self.rotary_emb = None  # Populated from teacher during model assembly

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_type = config.layer_types[i]
            if layer_type == "cs_lrad":
                block = LRADDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    chunk_size=config.chunk_size,
                    rank=config.rank,
                    dropout=config.dropout,
                    use_grad_checkpointing=config.use_grad_checkpointing,
                )
            elif layer_type == "transformer":
                block = nn.Identity()
            else:
                raise ValueError(f"Unsupported layer type '{layer_type}' at index {i}")
            self.layers.append(block)

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

        if past_key_values is not None and inputs_embeds.size(1) > 1:
            inputs_embeds = inputs_embeds[:, -1:, :]
            batch_size, seq_length, _ = inputs_embeds.shape

        h = self.drop(inputs_embeds)
        next_decoder_cache = [] if use_cache else None

        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                for idx, layer_past in enumerate(past_key_values):
                    if isinstance(layer_past, tuple) and len(layer_past) > 0 and layer_past[0] is not None:
                        p_item = layer_past[0]
                        while isinstance(p_item, (tuple, list)) and len(p_item) > 0:
                            p_item = p_item[0]
                        if isinstance(p_item, torch.Tensor):
                            past_length = p_item.size(-2)
                            break

            if attention_mask is not None and past_length == 0:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)
            else:
                position_ids = torch.arange(
                    past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0).expand(batch_size, -1)

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

            if "lrad" in cls_name:
                if is_gc:
                    def create_lrad_forward(module, p_state, u_cache, attn_mask):
                        def lrad_forward(x_in):
                            return module(x_in, past_state=p_state, use_cache=u_cache, attention_mask=attn_mask)
                        return lrad_forward

                    h, next_state = self._gradient_checkpointing_func(
                        create_lrad_forward(layer, past_state, use_cache, attention_mask),
                        h,
                    )
                else:
                    h, next_state = layer(
                        h, past_state=past_state, use_cache=use_cache, attention_mask=attention_mask
                    )
            elif isinstance(layer, nn.Identity) or cls_name == "identity":
                next_state = None
            else:
                # Native Teacher Transformer Layer (Qwen Decoder Layer)
                if is_gc:
                    def create_qwen_forward(module, attn_mask, pos_ids, p_kv, u_cache, pos_embeds):
                        def qwen_forward(x_in):
                            layer_outputs = module(
                                x_in,
                                attention_mask=attn_mask,
                                position_ids=pos_ids,
                                past_key_value=p_kv,
                                output_attentions=False,
                                use_cache=u_cache,
                                position_embeddings=pos_embeds,
                            )
                            if isinstance(layer_outputs, tuple):
                                h_res = layer_outputs[0]
                                ns = layer_outputs[1] if (u_cache and len(layer_outputs) > 1) else None
                            elif isinstance(layer_outputs, torch.Tensor):
                                h_res = layer_outputs
                                ns = None
                            else:
                                h_res = getattr(layer_outputs, "last_hidden_state", layer_outputs[0])
                                ns = getattr(layer_outputs, "past_key_values", None)
                            return h_res, ns
                        return qwen_forward

                    h, next_state = self._gradient_checkpointing_func(
                        create_qwen_forward(layer, attention_mask, position_ids, past_state, use_cache, position_embeddings),
                        h,
                    )
                else:
                    layer_outputs = layer(
                        h,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_state,
                        output_attentions=False,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                    if isinstance(layer_outputs, tuple):
                        h = layer_outputs[0]
                        next_state = layer_outputs[1] if (use_cache and len(layer_outputs) > 1) else None
                    elif isinstance(layer_outputs, torch.Tensor):
                        h = layer_outputs
                        next_state = None
                    else:
                        h = getattr(layer_outputs, "last_hidden_state", layer_outputs[0])
                        next_state = getattr(layer_outputs, "past_key_values", None)

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
        self.post_init()

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

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

        if num_logits_to_keep > 0 and hidden_states.size(1) > num_logits_to_keep:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]

        logits = self.lm_head(hidden_states)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        logits = torch.clamp(logits, min=-50.0, max=50.0)

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

        cache_position = kwargs.get("cache_position", None)
        if cache_position is not None and past_key_values is not None:
            position_ids = cache_position[-1:].unsqueeze(0) if cache_position.dim() == 1 else cache_position

        if position_ids is not None:
            position_ids = position_ids.to(input_ids.device)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None

        reordered_past = ()
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past += (None,)
            elif isinstance(layer_past, tuple):
                k, v = layer_past
                reordered_past += ((k.index_select(0, beam_idx), v.index_select(0, beam_idx)),)
            elif isinstance(layer_past, torch.Tensor):
                reordered_past += (layer_past.index_select(0, beam_idx),)
            else:
                reordered_past += (layer_past,)
        return reordered_past


# ==========================================
# Hugging Face Global Model Registration
# ==========================================

AutoModel.register(BareTorchConfig, BareTorchModel)
AutoModelForCausalLM.register(BareTorchConfig, BareTorchForCausalLM)
AutoModelForCausalLM.register(CSLRADConfig, CSLRADForCausalLM)