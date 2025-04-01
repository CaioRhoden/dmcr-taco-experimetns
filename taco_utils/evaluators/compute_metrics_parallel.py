import torch
from taco_utils.evaluators.metrics.testing_util import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Any, Dict
from datasets import load_dataset

TIMEOUT = 10


def check_correctness(args, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    result = []
    sample, generation = args
    _temp_run(sample, generation, debug, result)
    if result == []:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    
    return result[0]
def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = json.load(f)
        for _, res in enumerate(results):
            task_id = res['task_id']
            output = res['output']
            generations[task_id] = output
    return generations

def evaluate_generations(generations, samples, idx=None, debug=False):
    print("Evaluate Generations")
    assert len(generations.keys()) == len(samples)
    results = {}
    idx = 0
    for task_id, problem_generations in generations.items():
        
        sample = samples[idx]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(problem_generations):
            print(f"Code generation {o_idx} of {len(problem_generations)}")
            curr_res = [-2]
            try:
                curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
                if debug:
                    print(f"\nSuccessful compilation of task {o_idx}!")
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    if debug:
                        print(f"Results were not True for all test cases")
            except Exception as e:
                if debug:
                    print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
        results[task_id] = res
        idx += 1
    return results

def process_generation(args):
    task_id, sample, problem_generations, debug = args
    res = []
    debug = True
    import multiprocessing as mp
    check_args = [(sample, o) for o in problem_generations]
    with mp.Pool(mp.cpu_count()) as pool:
        results_list = pool.map(check_correctness, check_args)

    for curr_res in results_list:
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        if not np.all(curr_res):
            if debug:
                print(f"Results were not True for all test cases")
        assert isinstance(curr_res, list)
        res.append(curr_res)

    return task_id, res

def evaluate_generations_parallel(generations, samples, idx=None, debug=False):
    assert len(generations.keys()) == len(samples)
    task_id = list(generations.items())[0][0]
    gens = list(generations.items())[0][1]
    args = (task_id, samples[0], gens, debug)
    task_id, results_list = process_generation(args)
    results = {}
    results[task_id] = results_list
    return results

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
        

def compute_k_pass(results, k_list=[1, 10, 100]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen>0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    detail_metrics = {k:dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k

def calculate_1_pass(results: Dict[str, list], device="cuda:0") -> Dict[str, Any]:
    """Calculate 1-pass metrics for a given results dictionary"""
    metrics = {}
    task_ids = list(results.keys())
    for idx in task_ids:
        res = results[idx]
        max_length = max(len(inner) for inner in res)
        padded_res = [inner + [False] * (max_length - len(inner)) for inner in res]
        int_padded_res = [[int(item) for item in inner] for inner in padded_res]

        tensor = torch.tensor(int_padded_res, dtype=torch.int32, device=device)
        mask = (tensor == 1)
        tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
        sum_by_test: list[int] = tensor.sum(dim=0).cpu().numpy().tolist() # type: ignore
        metrics["tests"] = sum_by_test
        total_sum: int = sum(metrics["tests"])
        metrics[f"total"] = total_sum/(len(res)*max_length)
    
    return metrics


def compute_metrics_parallel(generation_file: str, taco, k_pass:list=[1, 10, 100], saving_file="taco_metrics.json"):
    # Initialize evaluation dataset with the same setup with generation
    # difficulties = ['ALL']
    # difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
    # skills = ['ALL']
    # skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

    # from datasets import load_dataset
    # taco = load_dataset('BAAI/TACO', split='test', difficulties=difficulties)
    # taco = load_dataset('BAAI/TACO', split='test', skills=skills)

    generations = load_generation(generation_file)


    results = evaluate_generations_parallel(generations, taco)
    # You can use evaluate_generations_parallel to parallel executing multiple outputs for each problem
    # results = evaluate_generations_parallel(generations, taco)
    metrics = {}
    metrics["k_pass"] = compute_k_pass(results, k_list=k_pass)
    metrics["normalized_sum"]= calculate_1_pass(results)
    metrics["results"] = results

    json.dump(metrics, open(saving_file, 'w'), indent=4)
    
