import os
import pandas as pd
from vlmeval.dataset import build_dataset
from vlmeval.config import supported_VLM
from vlmeval.inference import infer_data_job
from vlmeval.smp import get_pred_file_path

from vlmeval.vlm import BaseModel

DATASETS = [
    "RefCOCO",
    "GQA_TestDev_Balanced",
    "DocVQA_TEST",
    "ChartQA_TEST",
    "CountBenchQA",
    "POPE",
]

def evaluate_vlm(model, work_dir, nproc=1):
    """
    evaluate your vlm on refCOCO, refCOCO+, refCOCOg, GQA, DocVQA, ChartQA, CountBenchQA, and POPE.
    
    :param model: your model, specified using vlmeval.vlm.BaseModel class
    :param work_dir: directory to store the raw results in xlsx, and scores in json under work_dir/model_name/.

    note: the specific evals chosen are the same as those reported by moondream.ai for moondream3,
    which i am using as a reference to build VLMs.
    """

    model_name = "custom_model"
    if isinstance(model, str):
        model_name = model
        if model_name in supported_VLM:
            model : BaseModel = supported_VLM[model_name]()
        else:
            raise ValueError("The given model name is not supported by VLMEvalKit")
        
    elif hasattr(model, "model_name"):
        model_name = model.model_name
    
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
            api_nproc=1,
        )

        pred_file = get_pred_file_path(
            work_dir=work_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        res = dataset.evaluate(pred_file)
        results[dataset_name] = res

    return results