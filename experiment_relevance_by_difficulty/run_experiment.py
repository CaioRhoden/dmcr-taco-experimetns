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


# def process_data(
        
# )

def run():

    PATH  = "../data/TACO/processed"
    train = pl.read_ipc(f"{PATH}/train.feather")
    train_dict = load_from_disk("../data/TACO/train.hf")
    test_dict = load_from_disk("../data/TACO/test.hf")

    config = yaml.safe_load(open("config.yaml"))

    selected_problem = train.filter(pl.col("id") == 2545)
    prompt_input = selected_problem.select("input").to_struct().to_pandas().iloc[0]["input"]
    prompt = f"Please write a Python program \nQUESTION: \n{prompt_input} \n ANSWER: \n."


    wandb.init(
        project = "dmcr-taco-experiment-difficulty-relevance", 
        dir = "logs",
        id = f"2545", 
        name = f"2545",
        config = config,

    )


# run_inference(
#     prompt = prompt_input,
#     instruction = config["inference_configs"]["instruction"],
#     saving_path = f"{config['inference_configs']['saving_path']}/no_context.json",
#     model_path = config["inference_configs"]["model_path"],
#     model_configs = config["model_configs"],
#     num_returns = config["inference_configs"]["num_returns"],
#     num_generations = config["inference_configs"]["num_generations"],
#     log_datetime = config["inference_configs"]["log_datetime"],
#     quantization = config["inference_configs"]["quantization"]
    
# )

# parse_generations(
#     generations_path=f"{config['inference_configs']['saving_path']}/no_context.json",
#     id = 2545,
#     saving_path = f"{config['parse_configs']['saving_path']}/no_context_parsed.json"
# )

    evaluator = TACOEvaluator(
        generation_file = f"{config['parse_configs']['saving_path']}/no_context_parsed.json",
        taco = [train_dict[2545]],
        k_pass = [1, 10, 100],
        k_pass_path = f"{config['results_configs']['saving_path']}/no_context_1_pass.json",
        normalized_sum_path = f"{config['results_configs']['saving_path']}/no_context_normalized_sum.json"
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

