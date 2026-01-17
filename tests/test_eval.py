import pytest

def test_eval_on_smollm():
    from src.eval import evaluate_vlm
    from pprint import pprint
    model_name = "SmolVLM-256M"
    work_dir = "/vlm-project-volume/results"
    results = evaluate_vlm(model_name, work_dir, 1)
    pprint(results)

def test_eval_on_random_output():
    # from src.eval import evaluate_vlm
    # from src.model import CustomVLM
    # model = CustomVLM()
    pass
def test_eval_on_untrained_custom_model():
    pass