from typing import Any, Dict, Optional

import yaml
from torch.fx import GraphModule

import model_compressor  # isort:skip
from .utils import get_kwargs  # isort:skip


TARGET_MACHINE = 'RGDA0'
QLEVEL = 4

def _quantize(
    model: GraphModule,
    qparam_path: str,
    qformat_path: str,
    quantized_prefill: Optional[GraphModule] = None,
) -> GraphModule:
    return model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        decode_phase=quantized_prefill is not None,
        quantized_prefill_model=quantized_prefill,
        target_machine=TARGET_MACHINE,
        qlevel=QLEVEL,
        disable_auto_node_mapping=quantized_prefill is not None,
    )


def quantize_prefill_graph(
    model: GraphModule, qparam_path: str, qformat_path: str
) -> GraphModule:
    return _quantize(model, qparam_path, qformat_path)


def quantize_decode_graph(
    model: GraphModule,
    qparam_path: str,
    qformat_path: str,
    quantized_prefill: GraphModule,
) -> GraphModule:
    return _quantize(model, qparam_path, qformat_path, quantized_prefill)


def quantize_model(
    model: Dict[str, GraphModule],
    qparam_path: str,
    qformat_path: str,
) -> Dict[str, GraphModule]:
    
    quantized_prefill = quantize_prefill_graph(
        model=model["prefill"],
        qparam_path=qparam_path,
        qformat_path=qformat_path,
    )
    quantized_decode = quantize_decode_graph(
        model=model["decode"],
        qparam_path=qparam_path,
        qformat_path=qformat_path,
        quantized_prefill=quantized_prefill,
    )

    return {"prefill": quantized_prefill, "decode": quantized_decode}
