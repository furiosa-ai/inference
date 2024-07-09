import yaml
from torch.fx import GraphModule

import model_compressor  # isort:skip

from quantization.utils import get_kwargs  # isort:skip


def quantize_model(
    model: GraphModule, qparam_path: str, qformat_path: str
) -> GraphModule:

    model.config.use_cache = False

    return model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        target_machine='RGDA0',
        qlevel=4,
    )
