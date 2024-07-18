import argparse
import array
import os
from typing import Dict, List, Tuple

import mlperf_loadgen as lg
import torch
from backend_PyTorch import SUT_base as PyTorch_SUT_base
from furiosa_llm_models.gptj.symbolic.mlperf_submission import \
    GPTJForCausalLM as upstream_GPTJForCausalLM
from generator_RNGD import (MLPerfSubmissionBeamSearch,
                            expand_inputs_for_generation)
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.logits_process import \
    MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer
from transformers.utils.fx import get_concrete_args

from tests.e2e_pipe import LLMTestCase, prestep_furiosa_llm, Model
from furiosa_llm import LLMBackend, SamplingParams
from furiosa_llm.api import KvCacheSharingAcrossBeamsConfig
from tests.utils import PipelineParallelismMppp
from dataclasses import dataclass

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": int(
        os.environ.get("GPTJ_BEAM_SIZE", "4")
    ),  # only beam_size 4 is allowed for official submission
}

EARYLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 50256
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 30
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NUM_RETURN_SEQUENCES = 1
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.float32
QUANT_KV_DTYPE = torch.int8
BUCKET_SIZE = 2048
NUM_REAL_BATCH = 1

            
@dataclass
class GeneratorInputs:
    input_ids: List
    attention_mask: List


class SUT_base(PyTorch_SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        scenario,
        max_examples,
        use_gpu=False,
        network=None,
        qsl=None,
        args: argparse.Namespace = None,
    ):
        self.network = network
        self.model_name = "EleutherAI/gpt-j-6B"
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.dataset_path = dataset_path
        self.max_examples = max_examples
        self.scenario = scenario
        self.qsl = qsl
        print("Loading PyTorch model...")

        # dtype
        if dtype == "bfloat16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            print("BF16 autocast")
        elif dtype == "float16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32
        
        self.model = LLMTestCase(
            name="gpt-j-mlperf_submission-accuracy_test",
            model_metadata=Model.GPTJ_6B_28L_MLPERF_QUANTIZED,
            prompts=["dummy unused prompt"],
            sampling_params=SamplingParams(
                n=1, use_beam_search=True, best_of=4, max_tokens=128, min_tokens=30
            ),
            devices="npu:0:0-3, npu:0:0-3",
            mppp=PipelineParallelismMppp(),
            one_supertask_per_device=True,
            paged_attention_block_size=1,
            paged_attention_num_blocks=8192*2,
            prefill_buckets=[(1, 1920)],
            decode_buckets=[(4, 2048)],
            kv_cache_sharing_across_beams_config=KvCacheSharingAcrossBeamsConfig(
                4,
                128,
            ),
            use_blockwise_compile=True,
        )
        self.generator = prestep_furiosa_llm(self.model, backend=LLMBackend.FURIOSA_RT_V2)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1919,
            padding_side="left",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # construct SUT
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def inference_call(self, query, query_id=None):
        """Common for all scenarios"""
        torch_device_type = "cuda" if self.use_gpu else "cpu"

        input_ids_tensor = torch.tensor(query["input_ids_tensor"])
        input_masks_tensor = torch.tensor(query["input_masks_tensor"])

        # Moves the tensor to CPU or GPU as per argument passed by user
        input_ids_tensor = input_ids_tensor.to(torch_device_type)
        input_masks_tensor = input_masks_tensor.to(torch_device_type)

        with torch.inference_mode(), torch.autocast(
            device_type=torch_device_type,
            enabled=self.amp_enabled,
            dtype=self.amp_dtype if self.amp_enabled else None,
        ):
            input_batch = dict()
            input_batch["input_ids"] = input_ids_tensor
            input_batch["attention_mask"] = input_masks_tensor

            
            inputs = GeneratorInputs(input_ids=input_ids_tensor.tolist()[0], attention_mask=input_masks_tensor.tolist()[0])

            output = self.generator.engine.generate(inputs, sampling_params=self.model.sampling_params)
            output_batch = output.outputs[0].token_ids
            output_batch = torch.Tensor([output_batch]).to(torch.int64)

            input_batch_lengths = [x.shape[0] for x in input_batch["input_ids"]]

            output_batch_lengths = [x.shape[0] for x in output_batch]

            output_batch_truncated = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])

            output_batch_truncated = torch.stack(output_batch_truncated)

            # Loadgen monitors the reponse in corresponding functions
            if (
                self.scenario == "SingleStream" or self.scenario == "Server"
            ) and self.network == None:
                return output_batch_truncated

            pred_output_batch = output_batch_truncated.cpu().numpy()

            decoded_outputs = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in pred_output_batch
            ]
            response_text = decoded_outputs[0]

            # Loadgen monitors the response in GPT_QDL
            if self.network == "sut":
                return {
                    "pred_output_batch": pred_output_batch.tolist(),
                    "response_text": response_text,
                }

            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])


class SUT_Offline(SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        scenario,
        max_examples,
        use_gpu,
        network,
        qsl,
        args,
    ):
        SUT_base.__init__(
            self,
            model_path,
            dtype,
            dataset_path,
            scenario,
            max_examples,
            use_gpu,
            network,
            qsl,
            args,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Server(SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        scenario,
        max_examples,
        use_gpu,
        network,
        qsl,
        args,
    ):

        SUT_base.__init__(
            self,
            model_path,
            dtype,
            dataset_path,
            scenario,
            max_examples,
            use_gpu,
            network,
            qsl,
            args,
        )
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):
        # The issue queries function is called multiple times by the loadgen as per Poisson Distribution
        index = query_samples[0].index
        input_ids_tensor = self.qsl.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.qsl.data_object.source_encoded_attn_masks[index]
        text = self.qsl.data_object.sources[index]
        query = {
            "input_ids_tensor": input_ids_tensor.tolist(),
            "input_masks_tensor": input_masks_tensor.tolist(),
        }
        pred_output_batch = (
            self.inference_call(query, query_samples[0].id).cpu().numpy()
        )
        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)

        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        scenario,
        max_examples,
        use_gpu,
        network,
        qsl,
        args,
    ):
        SUT_base.__init__(
            self,
            model_path,
            dtype,
            dataset_path,
            scenario,
            max_examples,
            use_gpu,
            network,
            qsl,
            args,
        )
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):
        # This function is called by the loadgen after completing the previous query
        index = query_samples[0].index
        input_ids_tensor = self.qsl.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.qsl.data_object.source_encoded_attn_masks[index]
        query = {
            "input_ids_tensor": input_ids_tensor.tolist(),
            "input_masks_tensor": input_masks_tensor.tolist(),
        }

        pred_output_batch = (
            self.inference_call(query, query_samples[0].id).cpu().numpy()
        )

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)

        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


def get_SUT(
    model_path,
    scenario,
    dtype,
    dataset_path,
    max_examples,
    use_gpu=False,
    network=None,
    qsl=None,
    args: argparse.Namespace = None,
):
    if scenario == "Offline":
        return SUT_Offline(
            model_path,
            dtype,
            dataset_path,
            scenario,
            max_examples,
            use_gpu,
            network,
            qsl,
            args,
        )
    elif scenario == "Server":
        return SUT_Server(
            model_path,
            dtype,
            dataset_path,
            scenario,
            max_examples,
            use_gpu,
            network,
            qsl,
        )
    elif scenario == "SingleStream":
        return SUT_SingleStream(
            model_path,
            dtype,
            dataset_path,
            scenario,
            max_examples,
            use_gpu,
            network,
            qsl,
        )
