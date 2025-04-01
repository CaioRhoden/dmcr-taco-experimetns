from numpy import ndarray
from taco_utils.evaluators.compute_metrics import compute_metrics
from taco_utils.evaluators.compute_metrics_parallel import compute_metrics_parallel
import json

class TACOEvaluator():

    def __init__(self, generation_file: str, taco, k_pass:list, metrics_path:str):
        self.generation_file = generation_file
        self.taco = taco
        self.k_pass = k_pass
        self.metrics_path = metrics_path
    
    def evaluate(self) -> None:
        compute_metrics(self.generation_file, self.taco, k_pass=self.k_pass, saving_file=self.metrics_path)

    def evaluate_parallel(self) -> None:
        compute_metrics_parallel(self.generation_file, self.taco, k_pass=self.k_pass, saving_file=self.metrics_path)

    def extract_pass_1(self) -> float:
        with open(self.metrics_path) as f:
            metrics = json.load(f)
        return metrics["k_pass"]["pass@1"]
    
    def extracted_normalized_sum(self) -> float:
        with open(self.metrics_path) as f:
            metrics = json.load(f)
        return metrics["normalized_sum"]["total"]


