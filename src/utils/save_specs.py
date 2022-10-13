import json


def save_specs(json_path, json_obj) -> None:
    with open(json_path, 'w+') as f:
        json.dump(json_obj, f, indent=2)