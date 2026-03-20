import json 

def load_config(config_path):
    with open(config_path, "r") as fh:
        config = json.load(fh)

    return config