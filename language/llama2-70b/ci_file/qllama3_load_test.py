import yaml
from transformers import AutoConfig
import torch
from torch.utils.data import DataLoader
import json
import model_compressor

import joblib
import pickle

import argparse
from quantization.calibrate import load_pytorch_model
from transformers import LlamaConfig
from transformers import AutoTokenizer

from quantization.quantize import quantize_model
from RNGD_generator import MLPerfSubmissionGreedySearch
from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer
from furiosa_llm_models.llama3.symbolic.mlperf_submission_slice import LlamaForCausalLM


import furiosa_llm_models
import gc 
from torch.nn.functional import pad
import yaml 
import os
import accelerate

# Assume BLOCK_SIZE, NUM_BLOCKS, BUCKET_SIZE are fixed for now.
BLOCK_SIZE = 1
# bucket size would simply be a max value such as 2048 since we only provide one bucket
EARLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 2
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 1
NUM_BEAMS = 1
DO_SAMPLE = False
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.float32
QUANT_KV_DTYPE = torch.int8
BUCKET_SIZE = 2048

gen_kwargs = {
    "early_stopping": True,
    "min_new_tokens": 1,
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

DEVICE = 'cuda'

def make_calib_dataloader(tokenizer, batch_size=1):
    max_length = 2048
    prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    subfix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    sample_sentence = "Tell me the summary of a famous fable in no more than 5 sentences."
    input_token = tokenizer.encode(prefix+sample_sentence+subfix, add_special_tokens=True)  # 단일 문장 토큰화
    # padding size 계산
    padding_size = max_length - len(input_token)

    # 데이터 생성
    data_list = [
        {
            "input_ids": pad(torch.tensor(input_token, dtype=torch.int32), (padding_size, 0), value=2).view(1, -1).squeeze(0),
            "attention_mask": pad(torch.ones((1, len(input_token)), dtype=torch.int32), (padding_size, 0)).squeeze(0),
            "position_ids": pad(torch.arange(0, len(input_token), 1), (padding_size, 0)),
        }
    ]
    return DataLoader(data_list, batch_size=batch_size)

def perform_generation(
    generator,
    test_data_list,
    tokenizer,
    ref_path=None,
    res_path=None,
    config_dtype=None,
    update_gen_list=False,
    param_num='8b',
):
    if type(generator) == MLPerfSubmissionGreedySearch:  # mlperf submission generate
        # load reference generated tokens.
        update_ref_path = ref_path + f"/full_generated_data_list_{config_dtype}.json"
        # with open(update_ref_path, "r") as file:
        #     ref_data = json.load(file)

        results = []
        result_flag = True
        if update_gen_list:
            generated_data_list = []

    generation_output_dictionary = dict()
    with torch.no_grad():
        for idx, test_data in enumerate(test_data_list):
            if type(generator) == model_compressor.helper.QuantCausalLM:
                output = generator.generate(**test_data, **gen_kwargs)
            elif type(generator) == MLPerfSubmissionGreedySearch:
                # mlperf submission
                input_ids_tensor = []
                input_masks_tensor = []
                max_seq_len = 1024

                input_ids_tensor.append(
                    pad(
                        test_data["input_ids"],
                        (max_seq_len - test_data["input_ids"].shape[-1], 0, 0, 0),
                        value=tokenizer.pad_token_id,
                    )
                )

                input_masks_tensor.append(
                    pad(
                        test_data["attention_mask"],
                        (max_seq_len - test_data["attention_mask"].shape[-1], 0, 0, 0),
                        value=0,
                    )
                )

                input_ids_tensor = torch.cat(input_ids_tensor)
                input_ids_tensor = input_ids_tensor.to(torch.device('cuda:0'))
                input_masks_tensor = torch.cat(input_masks_tensor)
                input_masks_tensor = input_masks_tensor.to(torch.device('cuda:0'))

                logits_processor = LOGITS_PROCESSOR(
                    input_ids_tensor.shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
                )

                stopping_criteria = STOPPING_CRITERIA(
                    MAX_LENGTH,
                    None,
                )

                output = generator.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    max_length=MAX_LENGTH,
                    pad_token_id=PAD_TOKEN_ID,
                    eos_token_id=EOS_TOKEN_ID,
                    return_dict_in_generate=RETURN_DICT_IN_GENERATE,
                    kv_dtype=QUANT_KV_DTYPE,
                    bucket_size=BUCKET_SIZE,
                )

                generated_token = output[0][len(input_ids_tensor[0]) :]
                gen_sentence = tokenizer.decode(
                    generated_token, skip_special_tokens=True
                )
                if update_gen_list:
                    inp_decoded_text = tokenizer.decode(
                        input_ids_tensor[0], skip_special_tokens=True
                    )
                    generated_data = {
                        "inp_text": inp_decoded_text,
                        "gen_text": gen_sentence,
                    }
                    generated_data_list.append(generated_data)
                print(f"생성 토큰 문장 {idx}: {gen_sentence}")
                # compare submission model's decoded_test with reference sentences.
                # ref_sentence = ref_data[idx]["gen_text"]
                result_flag = check_diff(idx, ref_sentence, gen_sentence, results, result_flag)

            generation_output_dictionary[idx] = tokenizer.decode(
                output[0], skip_special_tokens=True
            )

        if (type(generator) == MLPerfSubmissionGreedySearch):  # mlperf submission generate
            # compare_results_path = res_path + f"/llama3.1-{param_num}_compare_result_{config_dtype}.json"
            # with open(compare_results_path, "w") as file:
            #     json.dump(results, file, indent=4)
            #     print(f"토큰 동치비교 결과가 저장되었습니다. dir: {compare_results_path}")
            if update_gen_list:
                with open(update_ref_path, "w") as file:
                    json.dump(generated_data_list, file, indent=4)
                print(
                    f"새로운 토큰 결과로 reference가 업데이트 되었습니다. dir: {update_ref_path}"
                )
            return result_flag


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to gpt-j model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--quant_data_path", help="path of submission quant data path")
    parser.add_argument("--generation_result_folder_path",help="path of the folder in which the log files of generation results are to be stored",)
    parser.add_argument("--ref_path", help="path of reference data")
    parser.add_argument("--res_path", help="path of ci result")
    parser.add_argument("--config_dtype", help="int8 or fp8")
    parser.add_argument("--param_num", default='8b', help="wheter to update gen_list")
    parser.add_argument("--update_gen_list", action="store_true", help="wheter to update gen_list")
    args = parser.parse_args()
    return args


def get_qlv4_load_models(
    model_path,
    output_path,
    qparam_out_path,
    qformat_out_path,
    prefill_exported_model_out_path,
    decode_exported_model_out_path,
):
    """
    Test 수행할 QLV4 quantized model 생성
    1. empty weight model 로딩
    2. qlv4 model로 변환
    3. qlv4 weight 로딩
    """
    CONFIG_PATH = os.path.join(model_path, "config.json")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
        custom_config = LlamaConfig.from_dict(config_dict)
        #custom_config.num_hidden_layers = 4
        
    with accelerate.init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(model_path)
            
    # device_map_path = "/home/home-mcl/phil/actions-runner/_work/data/furiosa_llm_modles_artifacts/quantized/meta-llama/Meta-Llama-3.1-8B-Instruct/mlperf_submission_slice/W8fA8fKV8f/32L/device_map.json"
    # with open(device_map_path, 'r') as file:
    #     device_map = json.load(file)
    
    # model = LlamaForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float32,
    #     config=custom_config,
    # )

    # if hasattr(model, 'hf_device_map'):
    #     model.device_map = model.hf_device_map
    #     model.module_name =  model.__class__.__module__ + "." + model.__class__.__name__

    # # Needs to place paged attention key value blocks on the same device as the transformer layers
    # if hasattr(model, "hf_device_map"):
    #     TRANSFORMER_LAYER_MODULE = "model.layers"  # valid only for LlamaForCausalLM
    #     device_map = {
    #         k.split(TRANSFORMER_LAYER_MODULE + ".")[1]: v
    #         for k, v in model.hf_device_map.items()
    #         if TRANSFORMER_LAYER_MODULE in k
    #     }
    # else:
    #     device_map = None
        

    model_type = type(model)
    traced_model = model.trace_all()
    prefill_model = traced_model['prefill']
    decode_model = traced_model['decode']

    test_prefill_quantized_model = model_compressor.create_quantsim_model(
        prefill_model,
        qformat_path=qformat_out_path,
        qparam_path=qparam_out_path,
        qlevel=4,
        target_machine='RGDA0',
        decode_phase=False,
        output_path=output_path,
    )

    test_decode_quantized_model = model_compressor.create_quantsim_model(
        decode_model,
        qformat_path=qformat_out_path,
        qparam_path=qparam_out_path,
        qlevel=4,
        target_machine='RGDA0',
        decode_phase=True,
        quantized_prefill_model=test_prefill_quantized_model,
        disable_auto_node_mapping=True,
    )

    # map_location = torch.device(DEVICE)
    
    # model_compressor.load(
    #     test_prefill_quantized_model, prefill_exported_model_out_path, map_location=map_location
    # )
    # model_compressor.load(
    #     test_decode_quantized_model, decode_exported_model_out_path, map_location=map_location
    # )
    # test_prefill_quantized_model = test_prefill_quantized_model.to(DEVICE)
    # test_decode_quantized_model = test_decode_quantized_model.to(DEVICE)

    # model_compressor.load(
    #     test_prefill_quantized_model, prefill_exported_model_out_path, enable_multi_gpu=True
    # )
    # model_compressor.load(
    #     test_decode_quantized_model, decode_exported_model_out_path, enable_multi_gpu=True
    # )

    # model_compressor.load(
    #     test_prefill_quantized_model, prefill_exported_model_out_path, map_location=map_location
    # )
    # model_compressor.load(
    #     test_decode_quantized_model, decode_exported_model_out_path, map_location=map_location
    # )    

    model_compressor.load(
        test_prefill_quantized_model, prefill_exported_model_out_path,
    )
    model_compressor.load(
        test_decode_quantized_model, decode_exported_model_out_path,
    )
    
    # device_map = test_prefill_quantized_model.device_map

    # return model_type, test_prefill_quantized_model, test_decode_quantized_model, device_map
    return model_type, test_prefill_quantized_model, test_decode_quantized_model



#load model_script
def create_qlv4_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,
        )
    
    test_data_list = make_calib_dataloader(tokenizer)

    qparam_path = os.path.join(args.quant_data_path, 'quant_param.npy')
    qformat_path = os.path.join(args.quant_data_path, 'quant_format.yaml')
    
    prefill_state_dict_path = os.path.join(args.quant_data_path, 'prefill.bin')
    decode_state_dict_path = os.path.join(args.quant_data_path, 'decode.bin')
    
    # model_type, prefill_quantized_model, decode_quantized_model, device_map = get_qlv4_load_models(
    model_type, prefill_quantized_model, decode_quantized_model = get_qlv4_load_models(
        args.model_path, 
        args.quant_data_path, 
        qparam_path, 
        qformat_path, 
        prefill_state_dict_path, 
        decode_state_dict_path
        )
    
    # quant_submission_model = {"prefill": prefill_quantized_model, "decode": decode_quantized_model}
    
    # submission_model_generator = MLPerfSubmissionGreedySearch(
    #     model=quant_submission_model, device_map=device_map
    # )

    # result_flag = perform_generation(
    #     submission_model_generator,
    #     test_data_list,
    #     tokenizer,
    #     ref_path=args.ref_path,
    #     res_path=args.res_path,
    #     config_dtype=args.config_dtype,
    #     update_gen_list=args.update_gen_list,
    #     param_num=args.param_num
    # )
    # print("----------------------------------------------")
    # print(f"토큰 동치 비교 결과 : {result_flag}")
    # print("----------------------------------------------")
    
    
if __name__ == "__main__":
    args = get_args()
    create_qlv4_model(args)
    print(f"llama3 qlv4 {args.config_dtype} load test is passed")
