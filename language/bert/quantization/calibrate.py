import argparse
import json
import pickle

import torch
import yaml
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForQuestionAnswering

import model_compressor  # isort:skip
# from model_compressor.utils import calib_generator
# from .calib_generator import calib_generator
# from transformers import AutoTokenizer

from .utils import get_kwargs, random_seed, set_optimization  # isort:skip

import os

BUCKET_SIZE = 384
PAD_TOKEN_ID = 0

def load_pytorch_model(model_path, model_config_path, use_gpu):
    with open(model_config_path) as f:
        config_json = json.load(f)

    config = BertConfig(**config_json)
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    model = BertForQuestionAnswering(config)
    model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    return model

def load_mlperf_submission_model(model_path, model_config_path, use_gpu):
    from furiosa_llm_models.bert.symbolic.mlperf_submission import BertForQuestionAnswering
    print("Loading BERT configs...")
    with open("bert_config.json") as f:
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

    dev = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("Loading PyTorch model...")
    model = BertForQuestionAnswering(config)
    model.to(dev)
    model.eval()
    model_file = os.environ.get(
        "ML_MODEL_FILE_WITH_PATH",
        "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch",
    )
    model.load_state_dict(torch.load(model_file), strict=False)

    return model


def cal_data_loader(data_path, batch_size, n_calib, model, generator):
    with open(data_path, "rb") as f:
        cal_features = pickle.load(f)

    data_list = [
        {
            "input_ids": torch.LongTensor(feature.input_ids),
            "attention_mask": torch.LongTensor(feature.input_mask),
            "token_type_ids": torch.LongTensor(feature.segment_ids),
        }
        for feature in cal_features[:n_calib]
    ]

    from RNGD_encoder import greedy_attention_packing_bert, bucket_pad
    from torch.nn.functional import pad

    for data in data_list:
        (
            input_ids,
            token_type_ids,
            attention_mask,
            position_ids,
            packed_target_locations,
        ) = greedy_attention_packing_bert(
            input_ids=bucket_pad(data["input_ids"].unsqueeze(0), BUCKET_SIZE),
            token_type_ids=bucket_pad(data["token_type_ids"].unsqueeze(0), BUCKET_SIZE),
            bucketized_attention_mask=bucket_pad(data["attention_mask"].unsqueeze(0), BUCKET_SIZE),
            pad_token_id=PAD_TOKEN_ID,
            compact_mask=False, # TODO
        )

        data.update(
            {
                "input_ids": input_ids[0],
                "token_type_ids": token_type_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
            }
        )


    # calib_dataloader = calib_generator(
    #                 data_path,
    #                 model,
    #                 'bert-base-uncased',
    #                 generator,
    #                 AutoTokenizer,
    #                 device=model.device,
    # )
    # return calib_dataloader
    # calib_dataloader = calib_generator(
    #     input_prompts,
    #     model,
    #     model_name,
    #     BertUnsplitPackedGenerator,
    #     AutoTokenizer,
    #     device,
    #     is_decode_only_model=False,
    # )

    return DataLoader(data_list, batch_size=batch_size)


def calibrate(model: GraphModule, qconfig, qparam_path, qformat_path, calib_dataloader):
    model.config.use_cache = False

    model = model_compressor.create_quantsim_model(
        model,
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    model_compressor.calibrate(
        model,
        calib_dataloader=calib_dataloader,
        **get_kwargs(model_compressor.calibrate, qconfig),
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
        kv_dtype=qconfig["kv_dtype"] if  "kv_dtype" in qconfig else 'bf16',
        disable_inout=(True, True),
    )

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch", "rngd"], default="pytorch", help="Backend"
    )
    parser.add_argument("--model_path", help="path to bert model")
    parser.add_argument("--model_source", help="bert model source")
    parser.add_argument("--model_config_path", help="path to bert model config")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibrated layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    parser.add_argument("--calib_data_path", help="path to calibration data")
    parser.add_argument(
        "--n_calib", type=int, default=-1, help="number of dataset to calibrate"
    )
    parser.add_argument(
        "--torch_numeric_optim",
        action="store_true",
        help="use PyTorch numerical optimizaiton for CUDA/cuDNN",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="use GPU instead of CPU for the inference"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.backend == "pytorch":
        if not args.gpu:
            raise ValueError(
                "Calibration on a device other than GPU is not supported yet."
            )
        model = load_pytorch_model(args.model_path, args.model_config_path, args.gpu)

    elif args.backend == "rngd":
        if not args.gpu:
            raise ValueError(
                "Calibration on a device other than GPU is not supported yet."
            )
        # from RNGD_SUT import get_rngd_sut
        # sut = get_rngd_sut(args)
        # model = sut.model
        model = load_mlperf_submission_model(args.model_path, args.model_config_path, args.gpu)
        model = model.trace()

        if args.model_source == 'mlperf_submission':
            from RNGD_encoder import BertMLPerfSubmissionEncoder
            generator = BertMLPerfSubmissionEncoder
        else:
            not NotImplementedError

    else:
        raise ValueError("Unsupported backend: {:}".format(args.backend))

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    dataloader = cal_data_loader(
        args.calib_data_path, qconfig["calib_batch_size"], args.n_calib, model, generator
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
