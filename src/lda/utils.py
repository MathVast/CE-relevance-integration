from dataclasses import dataclass
from typing import Dict, Union, List
from tqdm import tqdm
from datamaestro import prepare_dataset
from datamaestro_text.data.ir import TextItem
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
from src.utils import get_token_types_spans
from xpmir.learning.devices import DeviceInformation

import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class GetActivationsOutput:
    positive_activations: Dict
    negative_activations: Union[Dict, None]
    weak_negative_activations: Union[Dict, None]

def iterate_over_pairs_and_store_activations(
    dataset_name: str, 
    max_passages_per_query: int,
    pairs: Dict, 
    device: DeviceInformation,
    tokenizer: AutoTokenizer,
    extractor,
    target_module: str,
    layers_span: List[int],
    principal_components,
    top_eigen_vectors,
    input_parts: List[str],
    activation_fct,
    pred_position: int
):
    """Runs over a dictionnary in the following format: {qid1: [pid1, pid2, pid3, etc], etc} and gather the corresponding activations 
    before returning them.

    """
    global_storage = dict()
    passages = prepare_dataset(dataset_name)
    topics = prepare_dataset(dataset_name + '.queries')

    # We iterate over the dictionnary in the format : {qid: [pid1, pid2, pid3], etc} 
    for query_id in tqdm(pairs.keys()):
        count = 0
        query = topics.topic_ext(query_id)
        for assessment_id in pairs[query_id]:
            passage = passages.documents.document_ext(assessment_id)
            inputs = tokenizer(
                query[TextItem].text, 
                passage[TextItem].text,
                max_length=128, # True max length is 512 but it gets too long to compute
                truncation=True,
                padding="max_length", # To get every sample to the same size
                return_attention_mask=True,
                return_tensors="pt"
            ).to(device)
            inputs.requires_grad = False
            outputs = extractor(inputs)
            _ = activation_fct(outputs.logits, dim=1)[0][pred_position].detach().cpu().numpy()
            spans = get_token_types_spans(inputs["input_ids"], tokenizer)

            for layer_name, hidden_states in extractor.outputs_store.items():
                hidden_states = hidden_states.detach().cpu().numpy()
                layer_nb = int(layer_name.split(".")[3])
                if layer_nb not in layers_span:
                    continue
                module_name = ".".join(layer_name.split(".")[4:])
                if module_name == target_module:
                    for input_part in input_parts:
                        if input_part == "cls":
                            # If CLS, we pass it as is
                            token_activations = hidden_states[:,spans[0],:].squeeze(axis=0)
                        elif "query" in input_part:
                            # If query, we select the maximum number of query tokens we want to project and pass them as is
                            idx = int(input_part.split("_")[1])
                            token_activations = hidden_states[:,idx,:]
                        elif input_part == "document":
                            # If document, we average its activations
                            token_activations = np.mean(hidden_states[:,spans[3],:], axis=1)
                        else:
                            raise ValueError(f"Not recognized input part: {input_part}.")
                        
                        projections = token_activations @ top_eigen_vectors[f"layer_{layer_nb}"][input_part][principal_components].T
                        if f"layer_{layer_nb}" not in global_storage.keys():
                            global_storage[f"layer_{layer_nb}"] = dict()
                        if input_part not in global_storage[f"layer_{layer_nb}"].keys():
                            global_storage[f"layer_{layer_nb}"][input_part] = projections
                        else:
                            global_storage[f"layer_{layer_nb}"][input_part] = np.concatenate((global_storage[f"layer_{layer_nb}"][input_part], projections), axis=0)

            count += 1
            if count >= max_passages_per_query:
                break
                
    return global_storage
