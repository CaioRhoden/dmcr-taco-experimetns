
import re
import json

def parse_generations(generations: list, id: int, saving_path: str):
    

    ## Parsing python coding blocks starting with the ```python and ending with the ``` tag
    """
    Parse the generations of a task and save it to a JSON file.

    Args:
    generations (list): A list of dictionaries, each containing the generated text of a task.
    id (int): The task id.
    saving_path (str): The path where the parsed generation will be saved.

    Returns:
    None
    """
    
    gens = []
    for i in range(len(generations)):

        code_blocks = re.findall(r'```python(.*?)```', generations[i]["generated_text"], re.DOTALL)
        extracted_code = "\n".join([block.strip() for block in code_blocks])
        gens.append(extracted_code)
    
    results = [{
        "task_id": int(id),
        "output": gens
    }]


    ## Saving file
    try:
        json.dump(results, open(saving_path, "w"))
    except Exception as e:
        print("Error saving the generation file:", e)