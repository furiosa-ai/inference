import argparse
import array
import os
from typing import List

import mlperf_loadgen as lg
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tests.e2e_pipe import LLMTestCase, prestep_furiosa_llm, Model
from furiosa_llm import LLMBackend, SamplingParams
from furiosa_llm.api import KvCacheSharingAcrossBeamsConfig
from tests.utils import PipelineParallelismMppp
from dataclasses import dataclass
import json

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

BUCKET_SIZE = 2048
PREFILL_BUCKET_SIZE = BUCKET_SIZE - MAX_NEW_TOKENS
TOTAL_NUM_BLOCKS = PREFILL_BUCKET_SIZE + MAX_NEW_TOKENS * NUM_BEAMS
NUM_PADDING_BLOCKS = 1
NUM_REAL_BATCH = 1

            
@dataclass
class GeneratorInputs:
    input_ids: List
    attention_mask: List


class SUT_base:
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
        self.dump_path = args.dump_path
        if not self.dump_path.exists():
            with open(self.dump_path, "w") as f:
                json.dump([], f)
        self.dump = {}
        
        self.model = LLMTestCase(
            name="gpt-j-mlperf_submission-accuracy_test",
            model_metadata=Model.GPTJ_6B_28L_MLPERF_QUANTIZED,
            prompts=["dummy unused prompt"],
            sampling_params=SamplingParams(
                n=NUM_RETURN_SEQUENCES, use_beam_search=True, best_of=NUM_BEAMS, max_tokens=MAX_NEW_TOKENS, min_tokens=MIN_NEW_TOKENS
            ),
            devices=args.device,
            mppp=PipelineParallelismMppp(),
            one_supertask_per_device=True,
            paged_attention_block_size=1,
            paged_attention_num_blocks=8192*2, # TODO: TOTAL_NUM_BLOCKS * batch_size_in_decode + NUM_PADDING_BLOCKS, ex) (1920 + 128 * 4 + 1) * 1 = 2433
            prefill_buckets=[(1, PREFILL_BUCKET_SIZE)], # (1, 1920)
            decode_buckets=[(NUM_BEAMS * args.batch_size_in_decode, BUCKET_SIZE)], # (4 * 1, 2048) if batch_size_in_decode=1
            kv_cache_sharing_across_beams_config=KvCacheSharingAcrossBeamsConfig(
                NUM_BEAMS,
                MAX_NEW_TOKENS,
            ), # (4, 128)
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

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))
        # Pass each query to inference_call function
        # Activates only when scenario is Offline and network mode is None
        for i in tqdm(range(len(query_samples))):
            index = query_samples[i].index
            input_ids_tensor = self.qsl.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.qsl.data_object.source_encoded_attn_masks[index]
            text = self.qsl.data_object.sources[index]
            query = {
                "input_text": text,
                "input_ids_tensor": input_ids_tensor.tolist(),
                "input_masks_tensor": input_masks_tensor.tolist()
            }

            self.inference_call(query, query_samples[i].id)
            if self.dump_path:
                self.dump.update({"qsl_idx": index})
                self.dump.update({"input": query})
                self.dump.update({"output": self.response})
                with open(self.dump_path, "r") as f:
                    data = json.load(f)

                data.append(self.dump)
                data = sorted(data, key=lambda x: x["qsl_idx"])

                with open(self.dump_path, "w") as f:
                    json.dump(data, f)

    def inference_call(self, query, query_id=None):
        """Common for all scenarios"""
        torch_device_type = "cuda" if self.use_gpu else "cpu"

        input_ids_tensor = torch.tensor(query["input_ids_tensor"])
        input_masks_tensor = torch.tensor(query["input_masks_tensor"])

        # Moves the tensor to CPU or GPU as per argument passed by user
        input_ids_tensor = input_ids_tensor.to(torch_device_type)
        input_masks_tensor = input_masks_tensor.to(torch_device_type)

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

        self.response = {
                "pred_output_batch": pred_output_batch.tolist(),
                "response_text": response_text,
            }
        
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

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


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
