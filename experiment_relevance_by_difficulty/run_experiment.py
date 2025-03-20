import torch
import numpy as np
import random
import polars as pl
import json
from datasets import load_from_disk
import yaml
import wandb
import uuid

## Repo funtions
from taco_utils.evaluators.TACOEvaluator import TACOEvaluator
from taco_utils import run_inference, parse_generations

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
def generate_data_config(test: pl.DataFrame, train: pl.DataFrame):

    df = (
            test
            .filter(pl.col("tags") == "Probability")
            .group_by("difficulty")
            .agg(pl.col("id").sample(n=5, shuffle=True))
          )

    




def inference_id(run_id: int, context_type: str):

    PATH  = "../data/TACO/processed"
    train = pl.read_ipc(f"{PATH}/train.feather")
    test = pl.read_ipc(f"{PATH}/test.feather")
    train_solutions = pl.read_ipc(f"{PATH}/train_solutions.feather")
    train_dict = load_from_disk("../data/TACO/train.hf")
    test_dict = load_from_disk("../data/TACO/test.hf")

    try:
        config = yaml.safe_load(open("config.yaml"))
    except FileNotFoundError:
        print("Config .yaml not found, please create it")

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

    
    run_experiment(config, prompt, [test_dict[run_id]])


def run_experiment(config: dict, prompt: str, tests: list, context_type: str, run_id: int):


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
        "device": torch.cuda.get_device_name(i)
    })
    
    complete_log = wandb.Artifact(
        name = "complete_log",
        type = "log",
        description = "generatios, parse, metrics and input by run",   
    )
    complete_log.add_file(f"logs/{input_id}.txt")
    complete_log.add_file(f"{config['inference_configs']['saving_path']}/no_context.json")
    complete_log.add_file(f"{config['parse_configs']['saving_path']}/no_context_parsed.json")
    complete_log.add_file(f"{config['results_configs']['saving_path']}/no_context_1_pass.json")
    complete_log.add_file(f"{config['results_configs']['saving_path']}/no_context_normalized_sum.json")
    wandb.log_artifact(complete_log)

    wandb.finish()

def __main__():
    run()


if __name__ == "__main__":
    __main__()

