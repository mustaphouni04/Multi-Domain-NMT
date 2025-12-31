from huggingface_hub import HfApi
import os

def push_to_hub(repo_name: str, model_dir: str):
    api = HfApi()
    model = model_dir
    model_files = os.listdir(model)
    
    for file in model_files:
        if ".git" not in file:
            api.upload_file(
                path_or_fileobj=os.path.join(model,file),
                path_in_repo=file.split("/")[-1],
                repo_id=repo_name,
                repo_type="model",
            )
    return 0
