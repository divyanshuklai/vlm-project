import os
import sys
import argparse

import torch.distributed as dist
import torch

import pandas as pd

from vlmeval.dataset import build_dataset
from vlmeval.config import supported_VLM
from vlmeval.inference import infer_data_job
from vlmeval.smp import get_pred_file_path

from vlmeval.vlm import BaseModel

def setup_distributed():
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        dist.init_process_group(backend="nccl")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        print(f"Initialized process {rank}/{world_size}, Local rank {local_rank}")


DATASETS = [
    # "RefCOCO",
    # "GQA_TestDev_Balanced",
    # "DocVQA_TEST",
    # "ChartQA_TEST",
    # "CountBenchQA",
    "POPE",
]

def evaluate_vlm(model_ref : str, work_dir : str, nproc : int=1):
    """
    evaluate your vlm on refCOCO, refCOCO+, refCOCOg, GQA, DocVQA, ChartQA, CountBenchQA, and POPE.
    
    :param model: either the directory containing config.json and weights.pth or model name present in vlmeval.config.supportedVLM.
    The model must be inherited from vlmeval.vlm.BaseModel and must have implemented generate_inner.
    :param work_dir: directory to store the raw results in xlsx, and scores in json under work_dir/model_name/.

    note: the specific evals chosen are the same as those reported by moondream.ai for moondream3,
    which i am using as a reference to build VLMs.
    """

    model_name = "Custom Model"
    if os.path.isdir(model_ref):
        #model_ref is a path
        raise NotImplemented("custom model evals not implemented")
    elif model_ref in supported_VLM:
        #model_ref is supported by VLMEvalKit
        model_name = model_ref
        model = supported_VLM[model_ref]() #lazy load model
    else:
        raise ValueError("model not supported.")
    
    #create a sub-dir to save model results
    work_dir = os.path.join(work_dir, model_name)
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    
    results = {}

    for dataset_name in DATASETS:
        print(f"Evaluating {model_name} on {dataset_name}")

        dataset = build_dataset(dataset_name)
        if dataset is None:
            print(f"dataset {dataset_name} not found")
            continue
        
        infer_data_job(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            verbose=True,
            ignore_failed=True,
            api_nproc=nproc,
        )

        if dist.is_initialized():
            dist.barrier() # wait for all devices to finish this dataset

        if dist.is_initialized() and dist.get_rank() != 0:
            #only rank 0 device generates results. avoid race conditions.
            continue

        print(f"results generated for {model_name} on {dataset_name}, calculating scores...")

        pred_file = get_pred_file_path(
            work_dir=work_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        res = dataset.evaluate(pred_file)
        results[dataset_name] = res

    return results

def main():
    parser = argparse.ArgumentParser(description="evaluate your VLM on refCOCO/+/g, GQA, DocVQA, ChartQA, CountBenchQA, and POPE.")
    parser.add_argument("model", type=str, help="reference to the model path or model name supported by VLMEvalKit")
    parser.add_argument("work_dir", type=str, help="directory where results will be saved.")
    parser.add_argument("--nproc", type=int, help="processes per gpu to run. default 1", default=1)
    

    args = parser.parse_args()

    #for multi-gpu
    setup_distributed()

    model_ref = args.model
    work_dir = args.work_dir
    nproc = args.nproc
    evaluate_vlm(model_ref, work_dir, nproc)

    #cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

    