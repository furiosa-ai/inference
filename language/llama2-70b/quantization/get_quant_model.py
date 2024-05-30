import yaml
import os
import torch
from torch.utils.data import DataLoader
import model_compressor
from typing import Optional 
from dataset import Dataset
import copy
from furiosa_llm_models.llama.symbolic.huggingface_rope import LlamaForCausalLM
from accelerate import init_empty_weights
import gc

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False
}

# To use pipeline model
""" 
def generate_pipeline_model(model, quant_config, run_model_offloading, args, pass_weight_load=False):
    split_mode_kwargs = quant_config['split_mode']

    cache_ckpt_folder_path = args.model_path

    from .subgraph_utils import define_split_callback
    split_callback = define_split_callback(model)
    (
        subgraphs,
        subgraph_num_inputs,
        _,
    ) = model_compressor.multi_chip.split_into_subgraphs(model, split_callback=split_callback)
    model_config = model.config if hasattr(model, "config") else None
    model = model_compressor.multi_chip.pipeline_model(
        subgraphs, subgraph_num_inputs, model_config, run_model_offloading, cache_ckpt_folder_path, pass_weight_load=pass_weight_load,
    )

    return model
"""


def load_model_script(quant_config_path):
    with open(quant_config_path, 'r') as f:
        quant_config = yaml.safe_load(f)

    return quant_config


def get_quant_model(model, args):
    # Load model script and calibration dataloader (Refer to inference-compression/language/gpt-j/README.md on how to download evaluation and calibration dataset )
    quant_config = load_model_script(args.quant_config_path)
    
    model_type = type(model)

    (
        prefill_model,
        prefill_input_names,
        prefill_concrete_args,
    ) = model_compressor.helper.llama_custom_symbolic_trace(
        model, 
        input_names=["input_ids", "attention_mask", "position_ids"], 
        disable_check=True
    )
    
    prefill_quantized_model = model_compressor.create_quantsim_model(
        prefill_model,
        qformat_path=args.quant_format_path,
        qparam_path=args.quant_param_path,
        qlevel=quant_config["qlevel"],
        target_machine=quant_config["target_machine"],
        act_zp_equalizing=(quant_config["act_zp_equalizing"] if quant_config["act_zp_equalizing"] else 'disabled'),
        kv_dtype = quant_config["kv_dtype"] if "kv_dtype" in quant_config else 'bf16',
        disable_inout=(True, True),
        weighted_op_emul_dtype=args.weighted_op_emul_dtype,
        delete_org_weight=True,
        model_name='LlamaForCausalLM',
    )
    
    (
        decode_model,
        decode_input_names,
        decode_concrete_args,
    ) = model_compressor.helper.llama_custom_symbolic_trace(
        model,
        input_names=["input_ids", "past_key_values", "attention_mask", "position_ids"],
        disable_check=True,
    )

    decode_quantized_model = model_compressor.create_quantsim_model(
        decode_model,
        qformat_path=args.quant_format_path,
        qparam_path=args.quant_param_path,
        qlevel=quant_config["qlevel"],
        target_machine=quant_config["target_machine"],
        act_zp_equalizing=(quant_config["act_zp_equalizing"] if quant_config["act_zp_equalizing"] else 'disabled'),
        kv_dtype = quant_config["kv_dtype"] if "kv_dtype" in quant_config else 'bf16',
        disable_inout=(True, True),
        decode_phase=True,
        weighted_op_emul_dtype=args.weighted_op_emul_dtype,
        quantized_prefill_model=prefill_quantized_model,
        delete_org_weight=True,
        model_name='LlamaForCausalLM',
    )

    input_names = {
        "prefill_input_names" : prefill_input_names,
        "decode_input_names" : decode_input_names,
    }

    concrete_args = {
        "prefill_concrete_args": prefill_concrete_args,
        "decode_concrete_args": decode_concrete_args,
    }
    
    
    quant_models = {
        "prefill_model": prefill_quantized_model.eval(),
        "decode_model": decode_quantized_model.eval(),
    }
    
    #llama_causallm = model_compressor.helper.QuantCausalLM(quant_models, model_type, input_names, concrete_args)
    
    #return QuantPreAllocatedConcatGenerator(llama_causallm, bucket_size=2048)
    return model_compressor.helper.QuantCausalLM(quant_models, model_type, input_names, concrete_args)
