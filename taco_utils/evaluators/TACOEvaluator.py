from numpy import ndarray
from taco_utils.evaluators.compute_metrics import compute_key_pass
from taco_utils.evaluators.compute_normalized_sum_test_pass import compute_normalized_sum_test_pass
import json

class TACOEvaluator():

    def __init__(self, generation_file: str, taco, k_pass:list, k_pass_path: str, normalized_sum_path: str):
        self.generation_file = generation_file
        self.taco = taco
        self.k_pass = k_pass
        self.k_pass_path = k_pass_path
        self.normalized_sum_path = normalized_sum_path
    
    def evaluate(self) -> None:
        compute_key_pass(self.generation_file, self.taco, k_pass=self.k_pass, saving_file=self.k_pass_path)
        compute_normalized_sum_test_pass(self.generation_file, self.taco, saving_file=self.normalized_sum_path)

    def extract_pass_1(self) -> float:
        with open(self.k_pass_path) as f:
            metrics = json.load(f)
        return metrics["pass@1"]
    
    def extracted_normalized_sum(self) -> float:
        with open(self.normalized_sum_path) as f:
            metrics = json.load(f)
        return metrics["total"]


