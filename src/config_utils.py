from typing import Dict, List
from xpmir.papers.helpers import (
    configuration,
    LauncherSpecification
)
from xpmir.papers.helpers.optim import TransformerOptimization
from attrs import Factory, field


@configuration()
class Learner:
    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    checkpoint_listener_interval: int = field(default=100) # This one corresponds to the listeners that saves every checkpoints every n steps and store them
    checkpoint_interval: int = field(default=10) # This one corresponds to the basic mechanism that saves model every once in a while to avoid restarting from scratch
    sparsity_listener_interval: int = field(default=10) 

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    alpha: float = 0.01 # Controls the the regularization applied to the masks to sparsify them
    requirements: str = "duration=24h & cuda(mem=12G)"
    sample_rate: float = 1.0
    """Sample rate for triplets"""

    sample_max: int = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=20h & cpu(cores=8)"

@configuration()
class Retrieval:
    k: int = 1000
    batch_size: int = 512
    requirements: str = "duration=2 days & cuda(mem=24G)"

@configuration()
class Preprocessing(LauncherSpecification):
    requirements: str = "duration=12h & cpu(cores=4)"

@configuration()
class Postprocessing(LauncherSpecification):
    requirements: str = "duration=12h & cpu(cores=4)"

@configuration()
class GatherMatrices:
    # List of datasets to gather activations from. `max_passages_per_relevance_level` is the number of passages to gather for each relevance level
    # `relevance_levels` are the relevance levels where passages are considered relevant for their query
    list_of_datasets: List[Dict] = field(default=[{"dataset_name": "msmarco-passage.trec-dl-2019.judged", "max_passages_per_relevance_level": 20, "max_passages_per_query": 100, "relevance_levels": [2,3]}])

    max_queries: int = field(default=1000)
    max_query_len: int = 3

    # If set to true, we also gather activations for weak negatives and will include them in our analysis
    get_weak_negatives: bool = False

    indexation_requirements: str = "duration=2 days & cpu(cores=2)"
    sampling_requirements: str = "duration=2 days & cuda(mem=10G)"
    generation_requirements: str = "duration=4 days & cuda(mem=12G)"

    def check_datasets_list(self):
        for dataset in self.list_of_datasets:
            if "dataset_name" not in dataset and type(dataset.dataset_name) is not str:
                raise ValueError("Each dataset dictionary must contain a 'dataset_name' key whose value is a string.")
            if "max_passages_per_relevance_level" not in dataset and type(dataset.max_passages_per_relevance_level) is not int:
                raise ValueError("Each dataset dictionary must contain a 'max_passages_per_relevance_level' key whose value is an integer.")
            if "max_passages_per_query" not in dataset and type(dataset.max_passages_per_query) is not int:
                raise ValueError("Each dataset dictionary must contain a 'max_passages_per_query' key whose value is an integer.")
            if "relevance_levels" not in dataset and type(dataset.relevance_levels) is not list:
                raise ValueError("Each dataset dictionary must contain a 'relevance_levels' key whose value is a list of integers.")
        return True