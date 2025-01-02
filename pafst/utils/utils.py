
import json
import uuid
from pathlib import Path
from datetime import datetime

def write_json(path, data):
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(data, f)

def read_json(path):
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    return data

def mod_json(filepath, new_parent_dir):
    new_parent_dir=str(new_parent_dir)
    data=read_json(filepath)
    for d in data:
        new_spk_path=new_parent_dir + "/" + "/".join(d["speaker_path"].split("/")[-2:])
        new_sep_path=new_parent_dir + "/" + d["audio_filepath"].split("/")[-1]
        d["speaker_path"]=new_spk_path
        d["audio_filepath"]=new_sep_path
    write_json(filepath, data)
    return
    

def mk_temp_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex
    temp_dir = Path.cwd() / f"temp_dir_{timestamp}_{unique_id}"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir
