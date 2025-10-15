from typing import Dict, List
from xpmir.papers.helpers import (
    configuration,
    NeuralIRExperiment,
)
from attrs import Factory, field
from src.config_utils import GatherMatrices, Postprocessing
from transformers import AutoConfig

import logging
logging.basicConfig(level=logging.INFO)


@configuration()
class LDAStudy(NeuralIRExperiment):
    base_hf_id: str = "bert-base-uncased"
    ranker_id: str = "castorini/monobert-large-msmarco"
    list_of_datasets: List[Dict] = field(default=[{"dataset_name": "msmarco-passage.trec-dl-2019.judged", "max_passages_per_query": 20}])
    target_modules: List[str] = field(default=["output.dense"])
    start_layer: int = field(default=0)
    end_layer: int = field(default=24)
    max_query_len: int = 3
    gather_matrices: GatherMatrices = Factory(GatherMatrices)

    requirements: str = "duration=2d & cuda(mem=12G)"
    classification_requirements: str = "duration=8h & cuda(mem=12G)"
    post_processing: Postprocessing = Factory(Postprocessing)


    def check_datasets_list(self):
        for dataset in self.list_of_datasets:
            if "dataset_name" not in dataset and type(dataset.dataset_name) is not str:
                raise ValueError("Each dataset dictionary must contain a 'dataset_name' key whose value is a string.")
            if "max_passages_per_relevance_level" not in dataset and type(dataset.max_passages_per_relevance_level) is not int:
                raise ValueError("Each dataset dictionary must contain a 'max_passages_per_relevance_level' key whose value is an integer.")
        return True

    def check_end_layer(self):
        config = AutoConfig.from_pretrained(self.ranker_id)
        if self.end_layer > config.num_hidden_layers:
            logging.info("Specified end layer is greater than the number of hidden layers in the model. Setting end layer to the number of hidden layers in the model.")
            self.end_layer = config.num_hidden_layers