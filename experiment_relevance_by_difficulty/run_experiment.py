import torch
import numpy as np
import random
import polars as pl
import json
from datasets import load_from_disk
import yaml
import wandb
import uuid
import argparse

## Repo funtions
from taco_utils.evaluators.TACOEvaluator import TACOEvaluator
from taco_utils import run_inference, parse_generations
from datasets.arrow_dataset import Dataset, DatasetDict

seed = 42
# NumPy
np.random.seed(seed)
random.seed(seed)
pl.set_random_seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# split context data
from typing import Any, Dict, List

def run(data_path: str, config_path: str) -> None:
    """
    Run the experiment with specified data and configuration paths.

    Args:
        data_path (str): The path to the data configuration file.
        config_path (str): The path to the experiment configuration file.

    Returns:
        None
    """
    PATH = "../data/TACO/processed"
    train: pl.DataFrame = pl.read_ipc(f"{PATH}/train.feather")
    test: pl.DataFrame = pl.read_ipc(f"{PATH}/test.feather")
    train_solutions: pl.DataFrame = pl.read_ipc(f"{PATH}/train_solutions.feather")
    test_dict: Dataset | DatasetDict = load_from_disk("../data/TACO/test.hf")

    print("Splitting data to be used during experiment")
    generate_data_config(test, train, data_path)

    config: Dict[str, Any] = json.load(open(config_path))

    data: Dict[str, Any] = json.load(open(data_path))
    for context_type in ["no_context", "full_problem", "only_solutions"]:
        print(f"Running inference for context type: {context_type}")
        for difficulty in data["inputs"].keys():
            print(f"Running inference for difficulty: {difficulty}")
            for task_id in data["inputs"][difficulty]:
                run_inference_id(
                    run_id=task_id,
                    context_type=context_type,
                    difficulty=difficulty,
                    config=config,
                    test=test,
                    train=train,
                    train_solutions=train_solutions,
                    test_dict=test_dict
                )
                






def generate_data_config(test: pl.DataFrame, train: pl.DataFrame, path: str) -> None:
    """
    Generate a data partition configuration for the experiment.

    The function takes in the test and train dataframes, and returns a dictionary
    where the keys are the difficulty levels, and the values are dictionaries
    containing the task IDs as keys and the corresponding context task IDs as
    values.

    :param test: The test dataframe.
    :type test: pl.DataFrame
    :param train: The train dataframe.
    :type train: pl.DataFrame
    :return: A dictionary containing the data partition configuration.
    :rtype: Dict[str, Dict[int, List[int]]]
    """

    ### Sample 5 random examples from test as input by difficulty
    df = (
            test
            .filter(pl.col("tags") == "Probability")
            .group_by("difficulty")
            .agg(pl.col("id").sample(n=5, shuffle=True, with_replacement=False))
          )
    
    df.to_numpy()

    inputs = {}
    for _dif in df.to_numpy():
        inputs[_dif[0]] = _dif[1].tolist()

    ## Get 4 random examples from train set with the same diffulty from the input
    context = {}
    for _key in inputs.keys():
        for _id in inputs[_key]:
            _df = (
                train
                .filter(pl.col("tags") == "Probability")
                .filter(pl.col("difficulty") == _key)
                .sample(n=4, shuffle=True, with_replacement=False)
            )

            context[_id] = _df.select(pl.col("id")).to_numpy().squeeze(1).tolist()
    
    ## Save data partition to be used in running experiment
    with open(path, "w") as f:
        json.dump([input, context], f, ident=4)
    




def run_inference_id(
        run_id: int, 
        context_type: str, 
        difficulty: str,
        config: dict[str, Any],
        test: pl.DataFrame,
        train: pl.DataFrame,
        train_solutions: pl.DataFrame,
        test_dict: Dataset | DatasetDict

    ):

    

    selected_problem = test.filter(pl.col("id") == run_id)
    prompt_input = selected_problem.select("input").to_struct().to_pandas().iloc[0]["input"]
    prompt = f"Please write a Python program \nQUESTION: \n{prompt_input} \n ANSWER: \n."

    print(f"Context Type: {context_type}")


    ### SETUP CONTEXT BY TYPE
    match context_type:

        case "no_context":
            contexts = []

        case "full_problem":
            context_path = config["context"]
            contexts = json.load(open(context_path))
            context_ids = contexts[run_id]
            inputs = train.filter(pl.col("id").is_in(context_ids)).select("input").unique().to_dict()["input"]
            ## One solution per problem
            solutions = train_solutions.filter(pl.col("id").is_in(context_ids)).group_by(pl.col("id")).head(1).select("solution").unique().to_dict()["solution"]
            
            context_prompt = f"You will have to answer a programming quesiton in probability, we will pass before some examples of questions and solutions\n"
            for _idx in range(len(inputs)):
                context_prompt += f"EXAMPLE QUESTION {_idx}:\n {inputs[_idx]}\n EXAMPLE SOLUTION {_idx}:\n {solutions[_idx]}\n"
            
            prompt = context_prompt + prompt

        case "only_solutions":
            context_path = config["context"]
            contexts = json.load(open(context_path))

            context_ids = contexts[run_id]
            ## One solution per problem
            solutions = train_solutions.filter(pl.col("id").is_in(context_ids)).group_by(pl.col("id")).head(1).select("solution").unique().to_dict()["solution"]
            
            context_prompt = f"You will have to answer a programming quesiton in probability, we will pass before some examples of solutions for the same kind of problem\n"
            for _idx in range(len(solutions)):
                context_prompt += f"EXAMPLE SOLUTION {_idx}:\n {solutions[_idx]}\n"
            
            prompt = context_prompt + prompt


    inference(
        config, 
        prompt, 
        [test_dict[run_id]], 
        context_type=context_type, 
        run_id=run_id,
        diffiulty=difficulty
    )


def inference(
        config: dict, 
        prompt: str, 
        tests: list, 
        context_type: str, 
        run_id: int,
        diffiulty: str
        
        ):


    wandb.init(
        project = "dmcr-taco-experiment-difficulty-relevance", 
        dir = "logs",
        id = f"{run_id}_{context_type}", 
        name = f"{run_id}_{context_type}",
        config = config,

    )


    run_inference(
        prompt = prompt,
        instruction = config["inference_configs"]["instruction"],
        saving_path = f"{config['saving_paths']['inference'][context_type]}/{run_id}_inference.json",
        model_path = config["inference_configs"]["model_path"],
        model_configs = config["model_configs"],
        num_returns = config["inference_configs"]["num_returns"],
        num_generations = config["inference_configs"]["num_generations"],
        log_datetime = config["inference_configs"]["log_datetime"],
        quantization = config["inference_configs"]["quantization"]
        
    )

    parse_generations(
        generations_path=f"{config['saving_paths']['inference'][context_type]}/{run_id}_inference.json",
        id = 2545,
        saving_path = f"{config['saving_paths']['parsing'][context_type]}/{run_id}_parsing.json"
    )

    # tests here is the the list of the dict TACO of the sample being evaluated
    evaluator = TACOEvaluator(
        generation_file = f"{config['saving_paths']['parsing'][context_type]}/{run_id}_parsing.json",
        taco = tests,
        k_pass = [1, 10, 100],
        k_pass_path = f"{config['saving_paths']['results'][context_type]}/{run_id}_pass_k.json",
        normalized_sum_path = f"{config['saving_paths']['results'][context_type]}/{run_id}_normalized_sum.json"
    )

    input_id = str(uuid.uuid4())
    with open(f"logs/{input_id}.txt", "w") as f:
        f.write(prompt)




    # evaluator.evaluate()

    wandb.log({
        "pass@1": evaluator.extract_pass_1(),
        "normalized_sum": evaluator.extracted_normalized_sum(),
        "device": torch.cuda.get_device_name(i),
        "difficulty": diffiulty
    })
    
    complete_log = wandb.Artifact(
        name = "complete_log",
        type = "log",
        description = "generatios, parse, metrics and input by run",
    )
    complete_log.add_file(f"logs/{input_id}.txt")
    complete_log.add_file(f"{config['saving_paths']['inference'][context_type]}/{run_id}_inference.json")
    complete_log.add_file(f"{config['saving_paths']['parsing'][context_type]}/{run_id}_parsing.json")
    complete_log.add_file(f"{config['saving_paths']['results'][context_type]}/{run_id}_pass_k.json")
    complete_log.add_file(f"{config['saving_paths']['results'][context_type]}/{run_id}_normalized_sum.json")
    wandb.log_artifact(complete_log)

    wandb.finish()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    data_path = args.data_path
    config_path = args.config_path

    run(data_path, config_path)

