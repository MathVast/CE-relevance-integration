from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from ablations.config import AblationStudy
from ablations.utils import PerspectiveAblationStudy
from ablations.ablation import AblationAttention
from ablations.summary import SummaryAblationStudies, AgregationAblations
from utils import generic_dataset
from sampling import TopPassagesSampler

from datamaestro import prepare_dataset
from experimaestro.launcherfinder import find_launcher

from xpmir.rankers.standard import BM25
from functools import partial
import xpmir.interfaces.anserini as anserini
from xpmir.papers.helpers.samplers import ValidationSample
from transformers import AutoConfig


import logging

logging.basicConfig(level=logging.INFO)

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: AblationStudy
) -> PaperResults:
    # Verify that the end layer specified in the configuration isn't gretaer than the nb of layers
    # in the model
    cfg.check_end_layer()

    launcher_index = find_launcher(cfg.indexation_requirements)
    launcher_sampling = find_launcher(cfg.sampling_requirements)
    ablation_launcher = find_launcher(cfg.ablation_requirements)
    agregation_launcher = find_launcher(cfg.post_processing.requirements)

    basemodel = BM25.C()

    config = AutoConfig.from_pretrained(cfg.ranker_id)
    if config.num_labels == 1:
        target_position = 0
    else:
        target_position = 1


    retriever = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=basemodel,
    )

    cls_agregation_outputs = dict()
    cls_to_query_doc_agregation_outputs = dict()
    # More tests on the Doc
    doc_agregation_outputs = dict()
    doc_query_to_doc_query_agregation_outputs = dict()

    # More tests on the CLS
    cls_to_query_doc_sep_agregation_outputs = dict()
    cls_to_query_doc_both_sep_agregation_outputs = dict()

    no_interaction_agregation_outputs = dict()
    
    for dataset_name in cfg.list_of_datasets:
        # Load retriever, index, etc for this dataset
        dataset = prepare_dataset(dataset_name)

        sample_config = ValidationSample(size=cfg.max_queries_per_dataset) 
        datamaestro_dataset = generic_dataset(sample_config, dataset_name) # Randomly sample queries from the dataset to limit its size
        preprocessed_dataset = TopPassagesSampler.C(
            dataset=datamaestro_dataset,
            retriever=retriever(documents=dataset.documents, k=cfg.top_k),
        ).submit(launcher=launcher_sampling)


        # Get base values for this dataset
        PerspectiveAblationStudy.C(
            dataset_name=dataset_name,
            parsed_dataset=preprocessed_dataset,
        ).submit(launcher=ablation_launcher)
        
        cls_to_query_doc_ablations_outputs = list()
        cls_ablations_outputs = list()
        doc_ablation_outputs = list()
        doc_query_to_doc_query_ablations_outputs = list()
        cls_to_query_doc_sep_ablations_outputs = list()
        cls_to_query_doc_both_sep_ablations_outputs = list()
        no_interaction_ablations_outputs = list()
        for end_layer in range(cfg.start_layer, cfg.end_layer+1):
            cls_ablations_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["cls"],
                    cut_attention_to=[],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )

            cls_to_query_doc_ablations_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["cls"],
                    cut_attention_to=["query", "document"],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )

            doc_ablation_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["document"],
                    cut_attention_to=["document"],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )
            doc_query_to_doc_query_ablations_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["document", "query"],
                    cut_attention_to=["document", "query"],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )

            cls_to_query_doc_sep_ablations_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["cls"],
                    cut_attention_to=["query", "document", "sep_1"],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )
            cls_to_query_doc_both_sep_ablations_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["cls"],
                    cut_attention_to=["query", "document", "sep_1", "sep_2"],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )
            no_interaction_ablations_outputs.append(
                AblationAttention.C(
                    ranker_id=cfg.ranker_id,
                    base_hf_id=cfg.base_hf_id,
                    cut_attention_from=["query", "document", "sep_1", "sep_2"],
                    cut_attention_to=["query", "document", "sep_1", "sep_2"],
                    dataset_name=dataset_name,
                    parsed_dataset=preprocessed_dataset,
                    start_layer=cfg.start_layer,
                    end_layer=end_layer,
                    max_seq_length=cfg.max_seq_length,
                ).submit(launcher=ablation_launcher)
            )

        # We agregate PER DATASET first
        cls_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=cls_ablations_outputs
        ).submit(launcher=agregation_launcher)

        cls_to_query_doc_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=cls_to_query_doc_ablations_outputs
        ).submit(launcher=agregation_launcher)

        doc_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=doc_ablation_outputs
        ).submit(launcher=agregation_launcher)

        doc_query_to_doc_query_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=doc_query_to_doc_query_ablations_outputs
        ).submit(launcher=agregation_launcher)

        cls_to_query_doc_sep_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=cls_to_query_doc_sep_ablations_outputs
        ).submit(launcher=agregation_launcher)

        cls_to_query_doc_both_sep_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=cls_to_query_doc_both_sep_ablations_outputs
        ).submit(launcher=agregation_launcher)

        no_interaction_agregation_outputs[dataset_name] = AgregationAblations.C(
            dataset_name=dataset_name,
            target_label=target_position,
            ablations_output=no_interaction_ablations_outputs
        ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=cls_agregation_outputs,
    ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=cls_to_query_doc_agregation_outputs,
    ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=doc_agregation_outputs,
    ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=doc_query_to_doc_query_agregation_outputs,
    ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=cls_to_query_doc_sep_agregation_outputs,
    ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=cls_to_query_doc_both_sep_agregation_outputs,
    ).submit(launcher=agregation_launcher)

    SummaryAblationStudies.C(
        agregation_outputs=no_interaction_agregation_outputs,
    ).submit(launcher=agregation_launcher)
