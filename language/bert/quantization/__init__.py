import os

import torch
import yaml
from quantization.custom_symbolic_trace import custom_symbolic_trace

import model_compressor  # isort:skip


def quantize_model(model, qconfig_path, qparam_path, qformat_path):
    with open(qconfig_path, "r") as f:
        qconfig = yaml.safe_load(f)
    model, _, _ = custom_symbolic_trace(model)
    model.config.use_cache = False

    return model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        weight_calib_method=qconfig["weight_calib_method"],
        weight_granularity=qconfig["weight_granularity"],
        weight_dtype=qconfig["weight_dtype"],
        weight_nbits=qconfig["weight_nbits"],
        act_calib_method=qconfig["act_calib_method"],
        act_granularity=qconfig["act_granularity"],
        act_dtype=qconfig["act_dtype"],
        act_nbits=qconfig["act_nbits"],
        kv_dtype=qconfig["kv_dtype"] if "kv_dtype" in qconfig else "bf16",
        qlevel=qconfig["qlevel"],
        target_machine=qconfig["target_machine"],
        dataloader=None,
        disable_inout=(True, True),
    )
