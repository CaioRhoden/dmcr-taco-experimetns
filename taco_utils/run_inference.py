import datetime
from  dmcr.models import GenericInstructModelHF
import json



def run_inference(
        prompt: str,
        instruction: str,
        saving_path: str, 
        model_path: str,
        model_configs: dict,
        num_returns = 20,
        num_generations = 20,
        log_datetime = True,
        start_idx = 0,
        end_idx = 120,
        quantization = True
    ):

    ## Exception handling
    """
    Runs the LLM for the given prompt and instruction, and saves the result into a json file.
    
    Parameters:
    prompt (str): The prompt to be given to the model
    instruction (str): The instruction to be given to the model
    saving_path (str): The path where the generated text will be saved
    model_path (str): The path where the model is saved
    model_configs (dict): The configuration for the model
    num_returns (int): The number of texts to be generated
    num_generations (int): The number of times the model will be run
    max_length (int): The maximum length of the generated text
    log_datetime (bool): Whether to print the datetime when running the model
    
    Returns:
    None
    
    Raises:
    ValueError: If num_generations is greater than num_returns
    """
    if num_returns > num_generations:
        raise ValueError("num_generations should be less than num_returns")
    
    if num_generations > 1:
        model_configs["num_return_sequences"] = num_returns


    ## Running the model
    outputs = []
    llm = GenericInstructModelHF(model_path, quantization=quantization)
    for i in range(num_generations//num_returns):

        if log_datetime:
            print(f"Lopp {i}, {datetime.datetime.now()}")
        
        output = llm.run(prompt=prompt, instruction=instruction, config_params=model_configs)

        for res in output:
            outputs.append(res)
    

    ## Clean LLM from memory
    llm.delete_model()


    ## Saving file
    try:
        json.dump(outputs, open(saving_path, "w"))
    except Exception as e:
        print("Error saving the generation file:", e)