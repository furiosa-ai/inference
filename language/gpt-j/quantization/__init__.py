import inspect
from typing import Dict, Optional, Tuple

import torch
import yaml
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

import model_compressor  # isort:skip

from .custom_symbolic_trace import custom_symbolic_trace  # isort:skip
from .utils import get_kwargs  # isort:skip

SUPPORTED_CAUSAL_LM_ARCHITECTURES = ["GPTJForCausalLM"]


def quantize_model(model, qconfig_path, qparam_path, qformat_path):
    with open(qconfig_path, "r") as f:
        qconfig = yaml.safe_load(f)

    model, _, concrete_args = custom_symbolic_trace(model)

    model = model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    return build_causal_lm_for_graphmodule(
        graph_module=model,
        concrete_args=concrete_args,
    )


def build_causal_lm_for_graphmodule(
    graph_module: torch.fx.GraphModule, concrete_args: Dict
) -> PreTrainedModel:
    parent_class: PreTrainedModel = graph_module.class_for_deserialization

    if parent_class.__name__ not in SUPPORTED_CAUSAL_LM_ARCHITECTURES:
        raise ValueError(
            f"Unsupported Huggingface Transformers architecture: {parent_class.__name__}."
        )

    gm_forward = graph_module.forward
    sig_params = inspect.signature(gm_forward).parameters
    forward_input_names = [param.name for param in sig_params.values()]
    device = graph_module.device

    class GraphMouduleCausalForLM(parent_class):
        def __init__(self, config):
            super().__init__(config)
            self.init_past_key_values = [
                [torch.zeros(0).to(device), torch.zeros(0).to(device)]
            ] * self.config.n_layer

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
        ):
            if past_key_values is None:
                past_key_values = self.init_past_key_values

            forward_locals = locals()

            def _validate_input_arg(key):
                value = forward_locals[key]
                if value != concrete_args[key]:
                    raise ValueError(
                        f"The Custom Tracer has set '{key}' as {concrete_args[key]}, "
                        "but it's set as {value} during the forward pass. Please review the argument."
                    )

            _validate_input_arg("return_dict")
            _validate_input_arg("use_cache")
            _validate_input_arg("output_attentions")
            _validate_input_arg("output_hidden_states")

            forward_kwargs = {
                name: forward_locals[name] for name in forward_input_names
            }

            outputs = gm_forward(**forward_kwargs)

            if not return_dict:
                return outputs

            return CausalLMOutputWithPast(outputs)

    return GraphMouduleCausalForLM(config=graph_module.config)
