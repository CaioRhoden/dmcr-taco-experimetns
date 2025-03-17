from src.taco_evaluator.metrics.testing_util import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset
import torch

TIMEOUT = 10


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join()
    if p.is_alive():
        p.kill()
    if not result:
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
    assert len(generations.keys()) == len(samples)
    results = {}
    idx = 0
    for task_id, problem_generations in generations.items():
        sample = samples[idx]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(problem_generations):
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
    for o_idx, o in enumerate(problem_generations):
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
    return task_id, res

def evaluate_generations_parallel(generations, samples, idx=None, debug=False):
    assert len(generations.keys()) == len(samples)
    args = [(task_id, samples[i], problem_generations, debug) for i, (task_id, problem_generations) in enumerate(generations.items())]
    import multiprocessing as mp
    with mp.Pool(mp.cpu_count()) as pool:
        results_list = pool.map(process_generation, args)
    
    results = {task_id: res for task_id, res in results_list}
    return results

def calculate_1_pass(results: Dict[str, list], device="cuda:0"):
    """Calculate 1-pass metrics for a given results dictionary"""
    metrics = {}
    task_ids = list(results.keys())
    for idx in task_ids:
        res = results[idx]
        max_length = max(len(inner) for inner in res)
        padded_res = [inner + [False] * (max_length - len(inner)) for inner in res]
        tensor = torch.tensor(padded_res, dtype=torch.bool, device=device)
        metrics[idx] = tensor.sum(dim=0).cpu().numpy().tolist()
    
    return metrics
        



def compute_1_pass_by_test(generation_file: str, taco, debug=False, file="taco_1_pass_metrics.json", return_dict = False, return_results = False):
    # Initialize evaluation dataset with the same setup with generation
    # difficulties = ['ALL']
    # difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
    # skills = ['ALL']
    # skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

    # from datasets import load_dataset
    # taco = load_dataset('BAAI/TACO', split='test', difficulties=difficulties)
    # taco = load_dataset('BAAI/TACO', split='test', skills=skills)

    generations = load_generation(generation_file)

    results = evaluate_generations(generations, taco)
    # You can use evaluate_generations_parallel to parallel executing multiple outputs for each problem
    # results = evaluate_generations_parallel(generations, taco)
    metrics = calculate_1_pass(results)
     

    if not return_dict:
        json.dump(metrics, open(file, 'w'), indent=4)
    
    else:
        return metrics

    if return_results:
        return metrics, results
