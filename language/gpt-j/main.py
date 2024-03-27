import subprocess
import mlperf_loadgen as lg
import argparse
import os
import sys
from backend import get_SUT
sys.path.insert(0, os.getcwd())

import time
from datetime import timedelta
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline",
                        "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B", help="")
    parser.add_argument(
        "--dataset-path", type=Path, default="./data/cnn_eval.json", help="")
    parser.add_argument(
        "--calib-dataset-path", default="./data/cnn_dailymail_calibration.json", help="")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--dtype", default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true",
                        help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU instead of CPU for the inference")
    parser.add_argument("--audit_conf", default="audit.conf",
                        help="audit config for LoadGen settings during compliance runs")
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--quant_config_path", type=Path, help="a config for model quantization")
    parser.add_argument("--quant_param_path", type=Path, help="quantization parameters for calibraed layers")
    parser.add_argument("--quant_format_path", type=Path, help="quantization specifications for calibrated layers")
    parser.add_argument("--quantize", action="store_true", help="quantize model using ModelComPressor(MCP)")
    parser.add_argument('--torch_optim', default='default', type=str, choices=['default', 'none'], help='Torch optimization')
    parser.add_argument("--num_splits", type=int, default=1, help="")
    parser.add_argument("--split_idx", type=int, default=0, help="")
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def main():
    args = get_args()

    sut = get_SUT(
        model_path=args.model_path,
        scenario=args.scenario,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        max_examples=args.max_examples,
        use_gpu=args.gpu,
        num_splits=args.num_splits,
        split_idx=args.split_idx
    )

    if args.quantize:
        from quantization import quantize_model
        from quantization.utils import set_optimization, random_seed

        random_seed()
        set_optimization(args)

        sut.model = quantize_model(sut.model, args.quant_config_path, args.quant_param_path, args.quant_format_path)
    
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    log_path = os.environ.get("LOG_PATH")
    if not log_path:
        dataset_filename = args.dataset_path.stem
        quant_config_filename = args.quant_config_path.stem
        if args.quant_config_path is not None:
            if "fp32" in args.quant_config_path.as_posix() or args.quantize == False:
                if args.num_splits > 1:
                    log_path = f"build/logs/fp32/{dataset_filename}_{args.num_splits}_{args.split_idx}"
                else:
                    log_path = f"build/logs/fp32/{dataset_filename}"
            else:
                if args.num_splits > 1:
                    log_path = f"build/logs/{quant_config_filename}/{dataset_filename}_{args.num_splits}_{args.split_idx}"
                else:
                    log_path = f"build/logs/{quant_config_filename}/{dataset_filename}"
        else:
            log_path = f"build/logs/{dataset_filename}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True
    start_time = time.time()
    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings, args.audit_conf)

    end_time = time.time()
    time_seconds = end_time - start_time
    time_hour = timedelta(seconds=time_seconds)
    print(f"Test running time: {time_hour}")
    print("Test Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
