from typing import List
from xpmir.papers.helpers import (
    configuration,
    NeuralIRExperiment,
)
from xpmir.papers.helpers.samplers import ValidationSample
from attrs import Factory, field

from config_utils import Indexation, Learner, Postprocessing, Preprocessing, Retrieval

from transformers import AutoConfig
import logging
logging.basicConfig(level=logging.INFO)

@configuration()
class MaskLearning(NeuralIRExperiment):
    base_hf_id: str = "bert-base-uncased"
    ranker_id: str = "castorini/monobert-large-msmarco"
    
    pre_processing: Preprocessing = Factory(Preprocessing)
    sampling_requirements: str = "duration=1 days & cuda(mem=12G)"
    learner: Learner = Factory(Learner)
    start_layer: int = field(default=0)
    end_layer: int = field(default=24)
    max_seq_len: int = field(default=128)
    validation: ValidationSample = Factory(ValidationSample)
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    top_k: int = field(default=100)
    max_queries_per_dataset: int = field(default=1000)

    suffix: str = field(default="model.pth")
    test_datasets: List[str] = field(default=["irds.msmarco-passage.trec-dl-2019.judged"])
    test_threshold: float = field(default=0.99)
    test_requirements: str = "duration=24h & cuda(mem=12G)"
    post_processing: Postprocessing = Factory(Postprocessing)
    
    def check_end_layer(self):
        config = AutoConfig.from_pretrained(self.ranker_id)
        if self.end_layer > config.num_hidden_layers:
            logging.info("Specified end layer is greater than the number of hidden layers in the model. Setting end layer to the number of hidden layers in the model.")
            self.end_layer = config.num_hidden_layers