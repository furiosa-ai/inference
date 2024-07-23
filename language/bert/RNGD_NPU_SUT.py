import array
import json
import os
import sys
from typing import List

sys.path.insert(
    0,
    os.path.join(
        os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"
    ),
)
sys.path.insert(0, os.getcwd())

import json
from pathlib import Path
from dataclasses import dataclass

import mlperf_loadgen as lg
import numpy as np
import torch
import transformers
from furiosa_llm import LLMBackend, SamplingParams

from furiosa_llm_models.bert.symbolic.mlperf_submission import \
    BertForQuestionAnswering
from tests.e2e_pipe import LLMTestCase, Model, prestep_furiosa_llm
from tests.utils import PipelineParallelismMppp

from pytorch_SUT import BERT_PyTorch_SUT
from RNGD_encoder import BertMLPerfSubmissionEncoder, stack_tensors
from squad_QSL import get_squad_QSL
from torch.fx import GraphModule
from transformers import BertConfig

import tqdm

BUCKET_SIZE = 384
PAD_TOKEN_ID: int = 0  # EOS token

@dataclass
class EncoderInputs:
    input_ids: List
    attention_mask: List
    token_type_ids: List


class BERT_RNGD_NPU_SUT(BERT_PyTorch_SUT):
    def __init__(self, args):
        print("Loading BERT configs...")
        config_path = Path(__file__).parent.joinpath("bert_config.json")
        with open(config_path, "r") as f:
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

        self.network = args.network

        self.dump_path = args.dump_path
        # if not self.dump_path.exists():
        #     with open(self.dump_path, "w") as f:
        #         json.dump([], f)
        # self.dump = {}

        print("Loading PyTorch model...")
        self.model = LLMTestCase(
            name="mlperf-bert-submission-blockwise-accuracy_test",
            model_metadata=Model.BERT_LARGE_24L_MLPERF_QUANTIZED,
            prompts=[], # unused
            qa_context=[], # unused
            sampling_params=SamplingParams(), # unused
            devices="npu:0:0",
            mppp=PipelineParallelismMppp(),
            one_supertask_per_device=True,
            prefill_buckets=[
            (1, 384),
            (1, 192),
            (1, 272),
            (1, 128),
            (1, 160),
            (1, 320),
            (1, 144),
            (2, 96),
            ],
            # prefill_buckets=[(1, 384)],
            use_blockwise_compile=True,
        )

        self.encoder = prestep_furiosa_llm(self.model, backend=LLMBackend.FURIOSA_RT_V2)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        for i in tqdm.tqdm(range(len(query_samples)), unit="queries"):
            eval_features = self.qsl.get_features(query_samples[i].index)
            if self.dump_path:
                self.dump.update({"qsl_idx": query_samples[i].index})
            self.process_sample(eval_features, query_samples[i].id)

            if self.dump_path:
                with open(self.dump_path, "r") as f:
                    data = json.load(f)

                data.append(self.dump)
                data = sorted(data, key=lambda x: x["qsl_idx"])

                with open(self.dump_path, "w") as f:
                    json.dump(data, f)

    def process_sample(self, sample_input, query_id=None):
        # all: LLM_ENGINE_ARTIFACTS_PATH=/home/furiosa/llm_engine_artifacts/0722-mlperf-bert-submission-blockwise/
        # 384: LLM_ENGINE_ARTIFACTS_PATH=/home/furiosa/furiosa-ai/furiosa-runtime/.cache/accuracy_0721/test_mlperf_bert
        if self.network == "sut":
            input_ids = sample_input["input_ids"]
            input_mask = sample_input["input_mask"]
            segment_ids = sample_input["segment_ids"]
        else:
            input_ids = sample_input.input_ids
            input_mask = sample_input.input_mask
            segment_ids = sample_input.segment_ids

        query = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }

        if self.dump_path:
            self.dump.update({"input": query})
        # print(torch.count_nonzero(torch.LongTensor(input_ids)))
        # print(input_ids)
        # input_ids = input_ids[:171]
        # print(input_ids)
        # input_mask = input_mask[:171]
        # segment_ids = segment_ids[:171]
        # print(input_ids)
        # print(input_mask)
        # print(segment_ids)
        # print(input_ids)
        with torch.no_grad():
            inputs = EncoderInputs(input_ids=torch.IntTensor(input_ids).tolist(),
                attention_mask=torch.ByteTensor(input_mask).tolist(),
                token_type_ids=torch.IntTensor(segment_ids).tolist())
            # print(f"{input_ids=}")
            # print(f"{input_mask=}")
            # print(f"{segment_ids=}")
            model_output = self.encoder.engine.bert_unsplit_forward(inputs)
            # print("model_output", model_output)
            if self.dump_path:
                assert len(model_output) == 1
                self.dump.update({"output": {"output_ids": model_output[0].tolist()}})
            # print(input_ids)
            # print(len(input_ids))
            # print(model_output)
            # print(model_output.shape)
            # input_length = torch.LongTensor(input_ids).unsqueeze(0).shape[-1]
            # output = stack_tensors(model_output, max_shape=[input_length, 2])
            pad_val = -1.00000000e+10
            start_logits, end_logits = model_output[:, 0], model_output[:, 1]
            # start_logits[np.where(start_logits == pad_val)[0]] = np.iinfo(np.int32).min
            # end_logits[np.where(end_logits == pad_val)[0]] = np.iinfo(np.int32).min
            start_logits = np.ascontiguousarray(start_logits)
            end_logits = np.ascontiguousarray(end_logits)

            # start_logits = start_logits.squeeze(-1).contiguous()
            # end_logits = end_logits.squeeze(-1).contiguous()
            # start_logits = start_logits.cpu().numpy()[:, 0]
            # end_logits = end_logits.cpu().numpy()[:, 1]

            output = np.stack([start_logits, end_logits], axis=-1)
                # .squeeze(0)
                # .cpu()
                # .numpy()
            if self.network == "sut":
                return output.tolist()

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])


def get_rngd_npu_sut(args):
    return BERT_RNGD_NPU_SUT(args)
