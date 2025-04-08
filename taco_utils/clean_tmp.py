import os
import shutil

import os
import shutil

root_dir = os.getcwd()

def clean_directory():
# List all directories in root
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        # Check if it's a directory and starts with 'tmp'
        if os.path.isdir(item_path) and item.startswith("tmp"):
            print(f"Deleting: {item_path}")
            shutil.rmtree(item_path)  # Deletes directory and contents

    print("Done!")