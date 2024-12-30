import json

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def read_json(path):
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    return data
