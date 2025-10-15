from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from src.lda.config import LDAStudy
from src.lda.generate_data import gather_matrices
from src.lda.collect_directions import CollectDirections
from src.lda.classification import ComputeClassification, AggregateClassificationResults
from experimaestro.launcherfinder import find_launcher

import logging

logging.basicConfig(level=logging.INFO)

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: LDAStudy
) -> PaperResults:

    cfg.check_end_layer()

    output_agregation = gather_matrices(cfg)

    launcher_lda = find_launcher(cfg.requirements)
    launcher_classification = find_launcher(cfg.classification_requirements)
    launcher_postprocessing = find_launcher(cfg.post_processing.requirements)

    # 1: Aggregate the statistics across tokens and get the LDA
    collect_directions = CollectDirections.C(
        agregation_output=output_agregation,
        target_modules=cfg.target_modules,
        input_parts=["cls"] + [f"query_{i}" for i in range(cfg.max_query_len)] + ["document"],
        start_layer=cfg.start_layer,
        end_layer=cfg.end_layer,
    ).submit(launcher=launcher_lda)

    # 2: Project the activatiosn back onto these principal components
    output_classifications = list()
    for target_module in cfg.target_modules:
        output_classifications.append(
            ComputeClassification.C(
                agregation_output=output_agregation,
                list_of_datasets=cfg.list_of_datasets,
                ranker_id=cfg.ranker_id,
                base_hf_id=cfg.base_hf_id,
                collect_direction=collect_directions,
                input_parts=["cls", "document"],
                target_module=target_module,
                start_layer=cfg.start_layer,
                end_layer=cfg.end_layer,
            ).submit(launcher=launcher_classification)
        )

        output_classifications.append(
            ComputeClassification.C(
                agregation_output=output_agregation,
                list_of_datasets=cfg.list_of_datasets,
                ranker_id=cfg.ranker_id,
                base_hf_id=cfg.base_hf_id,
                collect_direction=collect_directions,
                input_parts=[f"query_{i}" for i in range(cfg.max_query_len)],
                target_module=target_module,
                start_layer=cfg.start_layer,
                end_layer=cfg.end_layer,
            ).submit(launcher=launcher_classification)
        )

    AggregateClassificationResults.C(list_of_classifications=output_classifications).submit(launcher=launcher_postprocessing)