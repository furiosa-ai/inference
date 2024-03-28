import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForQuestionAnswering

import model_compressor  # isort:skip

from quantization.custom_symbolic_trace import custom_symbolic_trace  # isort:skip
from quantization.utils import random_seed, set_optimization  # isort:skip


def load_pytorch_model(model_path, model_config_path, use_gpu):
    with open(model_config_path) as f:
        config_json = json.load(f)

    config = BertConfig(
        attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
        hidden_act=config_json["hidden_act"],
        hidden_dropout_prob=config_json["hidden_dropout_prob"],
        hidden_size=config_json["hidden_size"],
        initializer_range=config_json["initializer_range"],
        intermediate_size=config_json["intermediate_size"],
        max_position_embeddings=config_json["max_position_embeddings"],
        num_attention_heads=config_json["num_attention_heads"],
        num_hidden_layers=config_json["num_hidden_layers"],
        type_vocab_size=config_json["type_vocab_size"],
        vocab_size=config_json["vocab_size"],
    )

    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    model = BertForQuestionAnswering(config)
    model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


def cal_data_loader(data_path, batch_size, n_calib):
    with open(data_path, "rb") as f:
        cal_features = pickle.load(f)

    data_list = []
    for feature in cal_features:
        data_list.append(
            {
                "input_ids": torch.LongTensor(feature.input_ids),
                "attention_mask": torch.LongTensor(feature.input_mask),
                "token_type_ids": torch.LongTensor(feature.segment_ids),
            }
        )

    return DataLoader(data_list[:n_calib], batch_size=batch_size)


def calibrate(model, qconfig, qparam_path, qformat_path, calib_dataloader):
    model, _, _ = custom_symbolic_trace(model)
    model.config.use_cache = False

    model = model_compressor.create_quantsim_model(
        model,
        qformat_path=None,
        qparam_path=None,
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
        dataloader=calib_dataloader,
        disable_inout=(True, True),
    )

    model_compressor.calibrate(
        model,
        calib_dataloader=calib_dataloader,
        weight_calib_method=qconfig["weight_calib_method"],
        weight_granularity=qconfig["weight_granularity"],
        weight_dtype=qconfig["weight_dtype"],
        weight_nbits=qconfig["weight_nbits"],
        act_calib_method=qconfig["act_calib_method"],
        act_granularity=qconfig["act_granularity"],
        act_dtype=qconfig["act_dtype"],
        act_nbits=qconfig["act_nbits"],
        kv_dtype=qconfig["kv_dtype"] if "kv_dtype" in qconfig else "bf16",
        percentile=qconfig["percentile"],
        target_machine=qconfig["target_machine"],
    )

    model_compressor.save(
        model,
        qformat_out_path=qformat_path,
        qparam_out_path=qparam_path,
        weight_calib_method=qconfig["weight_calib_method"],
        weight_granularity=qconfig["weight_granularity"],
        weight_dtype=qconfig["weight_dtype"],
        weight_nbits=qconfig["weight_nbits"],
        act_calib_method=qconfig["act_calib_method"],
        act_granularity=qconfig["act_granularity"],
        act_dtype=qconfig["act_dtype"],
        act_nbits=qconfig["act_nbits"],
    )

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch"], default="pytorch", help="Backend"
    )
    parser.add_argument("--model_path", help="path to bert model")
    parser.add_argument("--model_config_path", help="path to bert model config")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    parser.add_argument("--calib_data_path", help="path to calibration data")
    parser.add_argument(
        "--n_calib", type=int, default=-1, help="number of dataset to calibrate"
    )
    parser.add_argument(
        "--torch_optim",
        default="default",
        type=str,
        choices=["default", "none"],
        help="Torch optimization",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="use GPU instead of CPU for the inference"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    sut = None

    if args.backend == "pytorch":
        if not args.gpu:
            raise ValueError(
                "Inference on a device other than GPU is not suppurted yet."
            )
        model = load_pytorch_model(args.model_path, args.model_config_path, args.gpu)

    else:
        raise ValueError("Unsupported backend: {:}".format(args.backend))

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    dataloader = cal_data_loader(
        args.calib_data_path, qconfig["calib_batch_size"], args.n_calib
    )
    calibrate(
        model,
        qconfig,
        args.quant_param_path,
        args.quant_format_path,
        dataloader,
    )


if __name__ == "__main__":
    main()
