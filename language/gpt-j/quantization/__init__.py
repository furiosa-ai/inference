from typing import Any, Dict, Tuple

import torch
import yaml
from transformers.generation.utils import inspect
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

import model_compressor  # isort:skip

from quantization.custom_symbolic_trace import custom_symbolic_trace  # isort:skip


def quantize_model(model, qconfig_path, qparam_path, qformat_path):
    with open(qconfig_path, "r") as f:
        model_script = yaml.safe_load(f)

    model_type = type(model)
    model, input_names, concrete_args = custom_symbolic_trace(model)

    model = model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        weight_calib_method=model_script["weight_calib_method"],
        weight_granularity=model_script["weight_granularity"],
        weight_dtype=model_script["weight_dtype"],
        weight_nbits=model_script["weight_nbits"],
        act_calib_method=model_script["act_calib_method"],
        act_granularity=model_script["act_granularity"],
        act_dtype=model_script["act_dtype"],
        act_nbits=model_script["act_nbits"],
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        act_zp_equalizing=(
            model_script["act_zp_equalizing"]
            if "act_zp_equalizing" in model_script
            else "disabled"
        ),
        dataloader=None,
        disable_inout=(True, True),
        kv_dtype=model_script["kv_dtype"] if "kv_dtype" in model_script else "bf16",
    )

    return QuantPreTrainedModel(model, model_type, input_names, concrete_args)


class QuantPreTrainedModel(PreTrainedModel):
    def __init__(self, quant_model, model_type, input_names, concrete_args):
        self.model_type = model_type
        super().__init__(quant_model.config)
        self.quant_model = quant_model
        self.config = quant_model.config
        self.input_names = input_names
        self.concrete_args = concrete_args

    def can_generate(self):
        return self.model_type.can_generate()

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(
            inspect.signature(self.model_type.prepare_inputs_for_generation).parameters
        )
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.model_type.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.model_type.prepare_inputs_for_generation(
            self, input_ids, **model_kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self.model_type._reorder_cache(past_key_values, beam_idx)

    def __call__(self, **kwargs):
        items_to_delete = []

        for key, value in kwargs.items():
            if (
                key in self.concrete_args
            ):  # check if the concrete args used when tracing and the elements of kwargs are equal
                if not value == self.concrete_args[key]:
                    raise ValueError(
                        f"The custom tracer set {key} as {self.concrete_args[key]} but kwargs sets {key} as {value}. Please check the argument again"
                    )
                items_to_delete.append(key)

        updated_kwargs = {
            key: value for key, value in kwargs.items() if key not in items_to_delete
        }

        if (
            "past_key_values" not in updated_kwargs.keys()
            or updated_kwargs["past_key_values"] == None
        ):  # add dummy past_key_valeus
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            updated_kwargs["past_key_values"] = tuple(
                [[torch.zeros(0).to(device), torch.zeros(0).to(device)]]
                * self.config.n_layer
            )

        return CausalLMOutputWithPast(self.quant_model(**updated_kwargs))
