from typing import Optional, Tuple, Union

import yaml
from torch.fx import GraphModule

import model_compressor  # isort:skip

from .utils import get_kwargs  # isort:skip


def quantize_model(
    model: GraphModule, qconfig_path: str, qparam_path: str, qformat_path: str
) -> GraphModule:
    with open(qconfig_path, "r") as f:
        qconfig = yaml.safe_load(f)

    quantized_model = model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    return quantized_model
