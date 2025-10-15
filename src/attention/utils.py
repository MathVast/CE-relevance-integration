
import json
import os

import logging

logging.basicConfig(level=logging.INFO)

def load_written_ids(jsonl_path):
    seen = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    seen.add((obj['pid'], obj['qid']))
                except Exception:
                    continue
    return seen

def open_attention_files(base_path, relevance_key, att_keys):
    files = {}
    for key in att_keys:
        path = os.path.join(base_path, f"{relevance_key}_attention_{key}.jsonl")
        files[key] = open(path, 'a', encoding='utf-8')
    return files

def serialize_slices_with_index(slices):
    return [{"start": s.start, "end": s.stop} for s in slices]

def write_data_line(f, pid, qid, spans): 
    data_dict = {
        "pid": pid,
        "qid": qid,
        "spans": serialize_slices_with_index(spans)
    }
    f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
