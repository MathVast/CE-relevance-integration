from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from src.attention.generation import GenerateAttentionPatterns, GenerateAttentionPatternsWeakNegatives
from src.attention.config import AttentionPatternStudy
from src.attention.summary import SummarizeAttentionPatterns
from experimaestro.launcherfinder import find_launcher
from xpmir.rankers.standard import BM25
from xpmir.papers.helpers.samplers import ValidationSample
from functools import partial
import xpmir.interfaces.anserini as anserini

from src.sampling import DiversePassagesSamplerWithHardNegatives
from src.utils import generic_dataset

import logging

logging.basicConfig(level=logging.INFO)

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: AttentionPatternStudy
) -> PaperResults:

    cfg.check_end_layer()

    """For every qrels in each of the datasets, gather the outputs from the attention and the MLP blocks (for each layer of the model)
    and aggregate them. The aggregation is done by summing the outer product of the activations, weighted by the final probability of relevance of the qrels,
    of the tokens of the same type.
    For CLS, SEP1, SEP2: Simple summation
    For Query's tokens: Simple summation but position-wise (over Q1, over Q2, etc. until Q_(max_query_len) )
    For Document's tokens: Simple summation of the average activation across all the document's tokens 
    """
    launcher_sampling =  find_launcher(cfg.gather_matrices.sampling_requirements)
    launcher_generation = find_launcher(cfg.gather_matrices.generation_requirements)
    launcher_indexation = find_launcher(cfg.gather_matrices.indexation_requirements)
    launcher_comparisons = find_launcher(cfg.post_processing.requirements)
    generation_outputs = list()

    # Define BM25 retriever
    base_model = BM25.C()
    index_builder = anserini.index_builder(launcher=launcher_indexation)

    retriever = partial(
        anserini.retriever,
        index_builder,
        model=base_model,
    )  #: Anserini based retrievers

    if cfg.gather_matrices.check_datasets_list():
        for dataset_config in cfg.gather_matrices.list_of_datasets:
            sample_config = ValidationSample(size=cfg.gather_matrices.max_queries) 
            datamaestro_dataset = generic_dataset(sample_config, dataset_config["dataset_name"]) # Randomly sample queries from the dataset to limit its size
            preprocessed_dataset = DiversePassagesSamplerWithHardNegatives.C(
                dataset=datamaestro_dataset,
                retriever=retriever(datamaestro_dataset.documents, k=200),
                relevance_levels=dataset_config["relevance_levels"],
                max_passages_per_relevance_level=dataset_config["max_passages_per_relevance_level"],
                max_passages_per_query=dataset_config["max_passages_per_query"],
            ).submit(launcher=launcher_sampling)

            generation_output = GenerateAttentionPatterns.C(
                ranker_id=cfg.ranker_id,
                base_hf_id=cfg.base_hf_id,
                dataset_name=dataset_config["dataset_name"],
                parsed_dataset=preprocessed_dataset,
                device=cfg.device,
            ).submit(launcher=launcher_generation)
            generation_outputs.append(generation_output)

            if cfg.gather_matrices.get_weak_negatives:
                generation_outputs.append(
                    GenerateAttentionPatternsWeakNegatives.C(
                        ranker_id=cfg.ranker_id,
                        base_hf_id=cfg.base_hf_id,
                        dataset_name=dataset_config["dataset_name"],
                        parsed_dataset=preprocessed_dataset,
                        target_count_per_query=dataset_config["max_passages_per_relevance_level"],
                        device=cfg.device,
                    ).submit(launcher=launcher_generation)
                )

    SummarizeAttentionPatterns.C(
        generation_outputs=generation_outputs,
        base_hf_id=cfg.base_hf_id,
        start_layer=cfg.start_layer,
        end_layer=cfg.end_layer,
    ).submit(launcher=launcher_comparisons)