from enum import Enum
from typing import Any
from experimaestro import Config

import torch
from transformers import AutoTokenizer
import numpy as np
from src.utils import split_list_by_values


class Baseline(Config):
    def generate_baseline(self, tokenizer: AutoTokenizer, input, model_embedding, device: torch.DeviceObjType) -> Any:
        pass

# Concrete strategies
class PadEverything(Baseline):
    def generate_baseline(self, tokenizer: AutoTokenizer, input, model_embedding, device: torch.DeviceObjType) -> Any:
        baseline = [tokenizer.pad_token_id for i in range(len(input[0]))]
        return model_embedding(torch.Tensor([baseline]).to(torch.int).to(device))

class PadQuery(Baseline):
    def generate_baseline(self, tokenizer: AutoTokenizer, input, model_embedding, device: torch.DeviceObjType) -> Any:
        baseline = list()
        input_splitted = split_list_by_values(np.array(input[0].cpu()), [tokenizer.sep_token_id])
        for token in input_splitted[0]:
            if token == tokenizer.cls_token_id:
                baseline.append(token)
            elif token == tokenizer.sep_token_id:
                baseline.append(token)
            else:
                baseline.append(tokenizer.pad_token_id)
        if len(input_splitted) == 3:
            baseline.extend(input_splitted[1]+input_splitted[2])
        else:
            baseline.extend(input_splitted[1])
        return model_embedding(torch.Tensor([baseline]).to(torch.int).to(device))

class PadQueryAndPassage(Baseline):
    def generate_baseline(self, tokenizer: AutoTokenizer, input, model_embedding, device: torch.DeviceObjType) -> Any:
        baseline = list()
        input_splitted = split_list_by_values(np.array(input[0].cpu()), [tokenizer.sep_token_id])
        for token in input_splitted[0]:
            if token == tokenizer.cls_token_id:
                baseline.append(token)
            elif token == tokenizer.sep_token_id:
                baseline.append(token)
            else:
                baseline.append(tokenizer.pad_token_id)
        for token in input_splitted[1]:
            if token == tokenizer.cls_token_id:
                baseline.append(token)
            elif token == tokenizer.sep_token_id:
                baseline.append(token)
            else:
                baseline.append(tokenizer.pad_token_id)
        return model_embedding(torch.Tensor([baseline]).to(torch.int).to(device))

class BaselineMethod(Enum):
    """Baseline methods for the attribution"""

    PAD_QUERY = "pad_query"

    PAD_QUERY_AND_PASSAGE = "pad_query_and_passage"

    PAD_EVERYTHING = "pad_everything"

    @staticmethod
    def values():
        return [e.value for e in BaselineMethod]

    def __str__(self):
        return self.value