from typing import List
from xpmir.papers.helpers import (
    configuration,
    NeuralIRExperiment,
)
from attrs import Factory, field

from src.config_utils import Postprocessing

from transformers import AutoConfig
import logging
logging.basicConfig(level=logging.INFO)

@configuration()
class AblationStudy(NeuralIRExperiment):
    base_hf_id: str = "bert-base-uncased"
    ranker_id: str = "castorini/monobert-large-msmarco"
    list_of_datasets: List[str] = field(default=["irds.msmarco-passage.trec-dl-2019.judged"])
    max_seq_length: int = field(default=128)
    start_layer: int = field(default=0)
    end_layer: int = field(default=24)

    top_k: int = field(default=100)
    max_queries_per_dataset: int = field(default=1000)

    indexation_requirements: str = "duration=2 days & cpu(cores=2)"
    sampling_requirements: str = "duration=1 days & cuda(mem=12G)"
    ablation_requirements: str = "duration=8h & cuda(mem=12G)"
    post_processing: Postprocessing = Factory(Postprocessing)

    def check_end_layer(self):
        config = AutoConfig.from_pretrained(self.ranker_id)
        if self.end_layer > config.num_hidden_layers:
            logging.info("Specified end layer is greater than the number of hidden layers in the model. Setting end layer to the number of hidden layers in the model.")
            self.end_layer = config.num_hidden_layers
