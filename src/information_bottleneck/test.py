from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from information_bottleneck.config import MaskLearning
from information_bottleneck.utils import AdvancedAblationTestFromPath, AgregationAdvancedAblationTests
from utils import generic_dataset
from sampling import TopPassagesSampler

from xpmir.learning.devices import CudaDevice
from functools import partial
from datamaestro import prepare_dataset
from pathlib import Path
from experimaestro.launcherfinder import find_launcher

from xpmir.papers.helpers.samplers import ValidationSample
from xpmir.rankers.standard import BM25
import xpmir.interfaces.anserini as anserini

from transformers import BertConfig

import logging

logging.basicConfig(level=logging.INFO)

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: MaskLearning
) -> PaperResults:
    cfg.check_end_layer()

    test_launcher = find_launcher(cfg.test_requirements)
    agregation_launcher = cfg.post_processing.launcher
    launcher_index = cfg.indexation.launcher
    launcher_sampling = find_launcher(cfg.sampling_requirements)

    advanced_test_outputs = list()

    basemodel = BM25.C()

    retriever = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=basemodel,
    )

    config = BertConfig.from_pretrained(cfg.ranker_id)
    if config.num_labels == 1:
        target_label = 0
    else:
        target_label = 1

    MASKING_STRATEGY = "gumbel_softmax"
    
    mask_folder = Path("/home/vast/CE-relevance-integration/src/information_bottleneck/masks/")
    for mask_file in mask_folder.rglob(cfg.suffix):
        for test_dataset in cfg.test_datasets:
            logging.info(f"Testing mask {cfg.suffix} on dataset {test_dataset}.")
            dataset = prepare_dataset(test_dataset)

            sample_config = ValidationSample(size=cfg.max_queries_per_dataset) 
            datamaestro_dataset = generic_dataset(sample_config, test_dataset) # Randomly sample queries from the dataset to limit its size
            preprocessed_dataset = TopPassagesSampler.C(
                dataset=datamaestro_dataset,
                retriever=retriever(documents=dataset.documents, k=cfg.top_k),
            ).submit(launcher=launcher_sampling)
            # Submit the task for testing the mask
            advanced_test_outputs.append(
                AdvancedAblationTestFromPath.C(
                    masking_strategy=MASKING_STRATEGY,
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    start_layer=cfg.start_layer,
                    end_layer=cfg.end_layer,
                    dataset_name=test_dataset,
                    parsed_dataset=preprocessed_dataset,
                    max_seq_len=cfg.max_seq_len,
                    mask_path=str(mask_file),
                    threshold=cfg.test_threshold,
                    device=CudaDevice.C()
                ).submit(launcher=test_launcher)
            )

    AgregationAdvancedAblationTests.C(
        advanced_ablation_outputs=advanced_test_outputs,
        target_label=target_label,
    ).submit(launcher=agregation_launcher)
