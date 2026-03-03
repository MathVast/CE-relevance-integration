from xpmir.papers.helpers import (
    configuration,
    NeuralIRExperiment,
)
from attrs import field
from transformers import AutoConfig
from attrs import Factory
from config_utils import GatherMatrices, Postprocessing

import logging
logging.basicConfig(level=logging.INFO)

@configuration()
class AttentionPatternStudy(NeuralIRExperiment):
    base_hf_id: str = "bert-base-uncased"
    ranker_id: str = "castorini/monobert-large-msmarco"
    """Identifier for the base model"""

    start_layer: int = field(default=0)
    end_layer: int = field(default=24)

    gather_matrices: GatherMatrices = Factory(GatherMatrices)
    """Configuration for gathering matrices"""
    start_layer: int = field(default=0)
    end_layer: int = field(default=24)
    post_processing: Postprocessing = Factory(Postprocessing)
    requirements: str = "duration=2d & cuda(mem=12G)"

    def check_end_layer(self):
        config = AutoConfig.from_pretrained(self.ranker_id)
        if self.end_layer > config.num_hidden_layers:
            logging.info("Specified end layer is greater than the number of hidden layers in the model. Setting end layer to the number of hidden layers in the model.")
            self.end_layer = config.num_hidden_layers
