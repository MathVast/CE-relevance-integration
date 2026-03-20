from functools import partial

from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.neural.huggingface import HFCrossScorer
from experimaestro.launcherfinder import find_launcher
import xpmir.interfaces.anserini as anserini
from xpmir.rankers.standard import BM25
from xpmir.papers.helpers.samplers import ValidationSample

from nig.attribution import AttributeInfoNCE
from nig.baseline import BaselineMethod, PadEverything, PadQuery, PadQueryAndPassage
from nig.config import NIGExperiment
from sampling import generic_dataset, AttributionInfoNCESampler
from nig.aggregation import AggregationResults
from nig.ablation import AblationSchemes

import logging
logging.basicConfig(level=logging.INFO)

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: NIGExperiment
) -> PaperResults:
    if not cfg.attribution.check_infonce():
        raise ValueError("One of the dataset's spec is missing an attribute. Please check the configuration file.")
    
    model : HFCrossScorer = HFCrossScorer.C(hf_id=cfg.base_hf_id, max_length=512).tag("model", cfg.base_hf_id)

    random = cfg.random
    launcher_attribution= find_launcher(cfg.attribution.requirements)
    launcher_aggregation = cfg.aggregation.preprocessing.launcher
    launcher_index = cfg.attribution.indexation.launcher
    launcher_ablation = find_launcher(cfg.ablation.requirements)

    attribution_results = list()
    base_model = BM25.C().tag("model", "bm25")

    retriever = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=base_model,
    )  #: Anserini based retrievers

    if cfg.attribution.baseline_method == str(BaselineMethod.PAD_QUERY):
        baseline_method = PadQuery.C()
    elif cfg.attribution.baseline_method == str(BaselineMethod.PAD_QUERY_AND_PASSAGE):
        baseline_method = PadQueryAndPassage.C()
    elif cfg.attribution.baseline_method == str(BaselineMethod.PAD_EVERYTHING):
        baseline_method = PadEverything.C()
    else:
        raise ValueError("The baseline method is not recognized")

    for dataset_spec in cfg.attribution.dataset_specs:
        sample_config = ValidationSample(size=dataset_spec["max_samples"]) 
        ds = generic_dataset(sample_config, dataset_spec["name"])
        attribution = AttributeInfoNCE.C(
            cross_scorer=model,
            sampler=AttributionInfoNCESampler.C(
                dataset=ds,
                retriever=retriever(ds.documents),
                relevant_levels=dataset_spec["relevance_levels"]
            ),
            baseline_method=baseline_method,
            batch_size=cfg.attribution.batch_size,
            steps=cfg.attribution.steps,
            random=random,
            max_samples=dataset_spec["max_samples"],
            max_input_length=cfg.attribution.max_input_length,
            device=cfg.device,
            checkpoint_interval=cfg.attribution.checkpoint_interval,
            compute_error=cfg.attribution.compute_error,
            adaptive_sampling=cfg.attribution.adaptive_sampling
        ).submit(launcher=launcher_attribution)
        attribution_results.append(attribution)

    aggregation_results = list()
    for attribution_result in attribution_results:
        aggregation_result = AggregationResults.C(
            cross_scorer=model,
            attribution_result=attribution_result,
            aggregate_per_token_type=cfg.aggregation.aggregate,
            device=cfg.device,
        ).submit(launcher=launcher_aggregation)
        aggregation_results.append(aggregation_result)

    ablation_scheme_outputs = AblationSchemes.C(
        target_modules_regex=cfg.ablation.target_modules_regex,
        pruning_percentages=cfg.ablation.pruning_percentages,
        aggregation_results=aggregation_results,
        ablation_specs=cfg.ablation.ablation_specs,
        aggregate_per_token_type=cfg.aggregation.aggregate,
    ).submit(launcher=launcher_ablation)