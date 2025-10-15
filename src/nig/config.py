from typing import Dict, List, Optional
from enum import Enum
from attrs import Factory, field

from xpmir.papers.helpers import (
    configuration,
    NeuralIRExperiment,
)

from src.config_utils import Indexation, Preprocessing
from src.nig.baseline import BaselineMethod

@configuration()
class Attribution:
    dataset_specs: List[Dict] = field(default=[{"name": "irds.beir.msmarco.test", "relevance_levels": [1], "max_samples": 5000, "target_label": 1}])
    baseline_method: str = field(default=BaselineMethod.PAD_QUERY)
    checkpoint_interval: int = 1
    compute_error: bool = False
    adaptive_sampling: bool = False
    max_input_length: int = 128
    steps: int = 50
    batch_size: int = 1
    requirements: str = "duration=4 days & cuda(mem=48G)"
    indexation: Indexation = Factory(Indexation)

    def check(self) -> bool:
        for dataset_spec in self.dataset_specs:
            if "name" not in dataset_spec.keys() or "relevance_levels" not in dataset_spec.keys() or "max_samples" not in dataset_spec.keys() or "target_label" not in dataset_spec.keys():
                return False
            else:
                return True
            
    def check_infonce(self) -> bool:
        for dataset_spec in self.dataset_specs:
            if "name" not in dataset_spec.keys() or "relevance_levels" not in dataset_spec.keys() or "max_samples" not in dataset_spec.keys():
                return False
            else:
                return True
            
@configuration()
class Aggregation:
    aggregate: bool = field(default=False)
    preprocessing: Preprocessing = Factory(Preprocessing)

class CombinationAblation(Enum):
    none = "NONE"
    intersection = "INTERSECTION"
    merge = "MERGE"

    def __str__(self):
        return self.name
    
@configuration()
class Ablation:
    ablation_specs: List[Dict] = field(default=[{CombinationAblation.none: {"name": "basic_msmarco","source_datasets": [{"name": "irds.beir.msmarco.test", "relevance_level": 1}]}}])
    pruning_percentages: List[float] = field(default=[0.00005, 0.0005, 0.005])
    target_modules_regex: Optional[List[str]] = field(default=None)
    requirements: str = "duration=4 days & cpu(cores=8)"

@configuration()
class NIGExperiment(NeuralIRExperiment):
    base_hf_id: str = "bert-base-uncased"
    """Identifier for the base model"""

    attribution: Attribution = Factory(Attribution)
    aggregation: Aggregation = Factory(Aggregation)
    ablation: Ablation = Factory(Ablation)
    metrics_processing: Preprocessing = Factory(Preprocessing)