import pytest
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def test_eval_on_smollm():
    from src.eval import evaluate_vlm
    from pprint import pprint
    model_name = "SmolVLM-256M"
    work_dir = "/results"
    results = evaluate_vlm(model_name, work_dir, 1)
    pprint(results)

def test_eval_on_random_output():
    # from src.eval import evaluate_vlm
    # from src.model import CustomVLM
    # model = CustomVLM()
    pass

def test_eval_on_untrained_custom_model():
    pass