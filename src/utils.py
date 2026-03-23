from typing import  Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from xpmir.utils.functools import cache as cache
from xpmir.papers.helpers.samplers import ValidationSample, prepare_collection, RandomFold
from datamaestro_text.data.ir import PairwiseSampleDataset
from datamaestro import prepare_dataset

import logging

logging.basicConfig(level=logging.INFO)

INPUT_PARTS = ["cls", "query_0", "query_1", "query_2", "sep_1", "document", "sep_2"]
INPUT_PART_TO_POSITION = {"cls": 0, "query": 1, "sep_1": 2, "document": 3, "sep_2": 4}
INTERESTING_LAYERS = ["attention_probs", "intermediate.dense"]
DATASET_NAME_TO_DOCUMENT_ID = {
    "irds.msmarco-passage.trec-dl-2019.judged": 'irds.msmarco-passage.documents@irds',
    "irds.msmarco-passage.trec-dl-2020.judged": 'irds.msmarco-passage.documents@irds', 
    "irds.msmarco-passage-v2.trec-dl-2021.judged": 'irds.msmarco-passage-v2.documents@irds', 
    "irds.msmarco-passage-v2.trec-dl-2022.judged": 'irds.msmarco-passage-v2.documents@irds',
    "irds.beir.fiqa.test": 'irds.beir.fiqa.documents@irds', 
    "irds.beir.trec-covid": 'irds.beir.trec-covid.documents@irds', 
    "irds.beir.nfcorpus.test": 'irds.beir.nfcorpus.documents@irds', 
    "irds.disks45.nocr.trec-robust-2004": 'irds.disks45.nocr.documents@irds', 
    "irds.dpr-w100.natural-questions.dev": 'irds.dpr-w100.documents@irds', # <-- Responsible for this mapping
     "irds.beir.bioasq.test": 'irds.beir.bioasq.documents@irds'
}

def filter_module(module: str, keywords: List[str]):
    return any(word in module for word in keywords)

def split_list_by_values(lst: List, values: List) -> List:
    """
    Split a list of integers based on specified values.

    :param List lst: List of integers to be split.
    :param List values: List of values to use as split points.
    :return List: A list of lists where the original list is split at each specified value.
    """
    result = []
    current_chunk = []

    for item in lst:
        current_chunk.append(item)
        if item in values:
            if current_chunk:
                result.append(current_chunk)
                current_chunk = []

    if current_chunk:
        result.append(current_chunk)

    return result

def get_token_types_spans(input, tokenizer) -> List[Tuple]:
    """Returns a list of spans corresponding to the positions in the input
    of respectively, the CLS token, the query's tokens, the document's tokens and the SEP tokens.

    """
    assert len(input.shape) == 2, "Input must be a 2D tensor"
    slices = [slice(0,1)] # Initiate the spans with the position of the CLS token
    input_splitted = split_list_by_values(np.array(input[0].cpu()), [tokenizer.sep_token_id])
    query_len = len(input_splitted[0]) - 2 # -1 to account for the CLS token and the first SEP token
    document_len = len(input_splitted[1]) - 1 # -1 to account for the second SEP token
    slices.append(slice(1, query_len + 1)) # The query's tokens
    slices.append(slice(query_len + 1, query_len + 2)) # The first SEP token
    slices.append(slice(query_len + 2, query_len + 2 + document_len)) # The document's tokens
    slices.append(slice(query_len + 2 + document_len, query_len + 2 + document_len + 1)) # The second SEP token
    return slices

def get_interesting_modules(model) -> List[str]:
    """
    Returns a list of the names of the interesting modules in the model.

    :return List[str]: List of module names.
    """
    interesting_layers = ["output.dense", "output.LayerNorm"]
    layer_names = list()
    for name, _ in model.named_modules():
        if any(word in name for word in interesting_layers):
            layer_names.append(name)

    return layer_names

def untuple(x):
    return x[0] if isinstance(x, tuple) else x

def extract_relevance_levels(qrels_metadata: Dict):
    levels = list()
    relevance = qrels_metadata.get('relevance', {})
    counts = relevance.get('counts_by_value', {})
    for key in counts.keys():
            try:
                level = int(key)
                if level >= 0:
                    # Returns only relevance levels corresponding to hard negatives or positives 
                    levels.append(level)
            except ValueError:
                continue
    
    return levels

def get_relevance_levels(dataset_name: str, sampled_passages: PairwiseSampleDataset)->Dict[str, Dict[str, int]]:
    """Get the relevance levels for each passage in the sampled dataset"""
    qrels = prepare_dataset(dataset_name + '.qrels')

    logging.info(f"Get all qrels.")
    reference_qrels = dict()
    for qrel in tqdm(qrels.iter()):
        reference_qrels[qrel.topic_id] = dict()
        for assessment in qrel.assessments:
            reference_qrels[qrel.topic_id][assessment.doc_id] = assessment.rel

    logging.info("Use the qrels to map the sampled passages to their relevance level.")
    dict_qrels = dict()
    for qrel in tqdm(sampled_passages.iter()):
        query_id = qrel.topics[0]
        dict_qrels[query_id] = dict()
        for passage in qrel.positives: # Called positives but actually contains all the passages
            if passage in reference_qrels[query_id]:
                dict_qrels[query_id][passage] = reference_qrels[query_id][passage]
            else:
                dict_qrels[query_id][passage] = 0
    return dict_qrels
    
def check_pid_and_qid(pid, qid, qrels):
    """ Used to check if a randomly sampled passage id belongs to the list of assessements for a query id.
    """
    for qrel in qrels.iter():
        if qrel.topic_id == qid:
            for assessment in qrel.assessments:
                if assessment.doc_id == pid:
                    return True
    return False

def is_pair_interesting(pair, regex: Optional[List[str]] = None):
    name, _ = pair
    if any(word in name for word in INTERESTING_LAYERS):
        if regex is not None:
            return filter_module(name, regex)
        else:
            return True
    else:
        return False

@cache
def generic_dataset(cfg: ValidationSample, dataset_name: str, launcher=None):
    """Sample dev topics to get a validation subset"""
    return RandomFold.C(
        dataset=prepare_collection(dataset_name),
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
        exclude=None,
    ).submit(launcher=launcher)
