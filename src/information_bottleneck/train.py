from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from information_bottleneck.config import MaskLearning
from information_bottleneck.utils import (
    DistillationPairwiseTrainerWithSparsification, 
    GumbelSigmoidMaskingStrategy, 
    GumbelSoftmaxMaskingStrategy, 
    MainListener, 
    LearningMaskForCrossScorer, 
    MaskingStrategyEnum, 
    SigmoidMaskingStrategy, 
)

from transformers import AutoConfig
from xpmir.learning.devices import DEFAULT_DEVICE, Device
from xpmir.learning.optim import AdamW, ParameterOptimizer
from xpmir.learning.schedulers import LinearWithWarmup
from experimaestro import tag
from experimaestro.launcherfinder import find_launcher
from utils import INPUT_PART_TO_POSITION

## Imports for the optimization: ##
from xpmir.papers.helpers.samplers import (
    msmarco_v1_docpairs_efficient_sampler,
    msmarco_v1_validation_dataset,
    prepare_collection,
)
from xpmir.rankers.standard import BM25
import xpmir.interfaces.anserini as anserini
from xpmir.rankers import scorer_retriever, ScorerOutputType
from functools import partial
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.distributed import DistributedHook
import xpmir.letor.distillation.pairwise as distillation_pairwise
from xpmir.learning.learner import Learner
from experimaestro import setmeta
from xpmir.learning.optim import (
    TensorboardService,
    ModuleLoader,
)
from xpmir.letor.learner import ValidationListener

import logging

logging.basicConfig(level=logging.INFO)

RANGE_ALPHA = [1e-3, 1e-2, 1e-1, 1, 10] # [1e-3, 1e-2, 1e-1, 1, 10]
RANGE_LR = [1e-1, 1] # [1e-1, 1]
MASKING_STRATEGIES = ["gumbel_softmax"]
RANGE_MASKS_INITIAL_VALUES = [3.0]
RANGE_TAU = [1.0]

def advanced_ablation_optimization(
    cfg: MaskLearning, 
    ranker_id: str, 
    base_hf_id: str,
    tensorboard_service: TensorboardService, 
    device: Device = DEFAULT_DEVICE, 
    random: bool = False,
)-> ModuleLoader:
    optimization_launcher = find_launcher(cfg.learner.requirements)
    launcher_index = cfg.indexation.launcher
    launcher_ri_preprocessing = cfg.pre_processing.launcher

    documents = prepare_collection("irds.msmarco-passage.documents")
    ds_val = msmarco_v1_validation_dataset(
        cfg.validation, launcher=launcher_ri_preprocessing
    )

    base_model = BM25.C().tag("model", "bm25")

    retrievers = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=base_model,
    )  #: Anserini based retrievers

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
        device=device,
    )  #: Model-based retrievers

    val_retrievers = partial(
        retrievers, store=documents, k=cfg.learner.validation_top_k
    )

    ranker_config = AutoConfig.from_pretrained(ranker_id)

    outputs = list()
    for masking_strategy_text in MASKING_STRATEGIES:
        for masks_initial_value in RANGE_MASKS_INITIAL_VALUES:
            for tau in RANGE_TAU:
                if masking_strategy_text == str(MaskingStrategyEnum.SIGMOID):
                    masking_strategy = SigmoidMaskingStrategy.C(
                        num_categories=len(INPUT_PART_TO_POSITION.keys()), 
                        num_attention_heads=ranker_config.num_attention_heads, 
                        start_layer=cfg.start_layer, 
                        end_layer=cfg.end_layer, 
                        max_seq_len=cfg.max_seq_len, 
                        masks_initial_value=tag(masks_initial_value)
                    )
                elif masking_strategy_text ==  str(MaskingStrategyEnum.GUMBEL_SOFTMAX):
                    masking_strategy = GumbelSoftmaxMaskingStrategy.C(
                        num_categories=len(INPUT_PART_TO_POSITION.keys()), 
                        num_attention_heads=ranker_config.num_attention_heads, 
                        start_layer=cfg.start_layer, 
                        end_layer=cfg.end_layer, 
                        max_seq_len=cfg.max_seq_len, 
                        masks_initial_value=tag(masks_initial_value),
                        tau=tag(tau)
                    )
                elif masking_strategy_text == str(MaskingStrategyEnum.GUMBEL_SIGMOID):
                    masking_strategy = GumbelSigmoidMaskingStrategy.C(
                        num_categories=len(INPUT_PART_TO_POSITION.keys()), 
                        num_attention_heads=ranker_config.num_attention_heads, 
                        start_layer=cfg.start_layer, 
                        end_layer=cfg.end_layer, 
                        max_seq_len=cfg.max_seq_len, 
                        masks_initial_value=tag(masks_initial_value),
                        tau=tag(tau)
                    )
                else:
                    raise ValueError("Masking strategy not recognized.")
                
                masked_scorer = LearningMaskForCrossScorer.C(
                    outputType=ScorerOutputType.PROBABILITY,
                    ranker_id=ranker_id,
                    base_model_id=base_hf_id,
                    start_layer=cfg.start_layer,
                    end_layer=cfg.end_layer,
                    max_seq_len=cfg.max_seq_len,
                    masking_strategy=masking_strategy.tag("masking", masking_strategy_text),
                )

                for alpha in RANGE_ALPHA:
                    monobert_trainer = DistillationPairwiseTrainerWithSparsification.C(
                        alpha=tag(alpha),
                        # lossfn=pairwise.PointwiseCrossEntropyLoss(),
                        lossfn=distillation_pairwise.MSEDifferenceLoss.C(),
                        sampler=msmarco_v1_docpairs_efficient_sampler(
                            sample_rate=cfg.learner.sample_rate,
                            sample_max=cfg.learner.sample_max,
                            launcher=launcher_ri_preprocessing,
                        ),
                        batcher=PowerAdaptativeBatcher.C(),
                        batch_size=cfg.learner.optimization.batch_size,
                        hooks=[]
                    )

                    for lr in RANGE_LR:
                        scheduler = (
                            LinearWithWarmup.C(
                                num_warmup_steps=cfg.learner.optimization.num_warmup_steps,
                                min_factor=cfg.learner.optimization.warmup_min_factor,
                            )
                            if cfg.learner.optimization.scheduler
                            else None
                        )

                        optimizer = [
                            ParameterOptimizer.C(
                                scheduler=scheduler,
                                optimizer=AdamW.C(
                                        lr=tag(lr),
                                        weight_decay=cfg.learner.optimization.weight_decay,
                                        eps=cfg.learner.optimization.eps,
                                    )
                            ),
                        ]

                        validation = ValidationListener.C(
                            id="bestval",
                            dataset=ds_val,
                            retriever=model_based_retrievers(
                                documents,
                                retrievers=val_retrievers,
                                scorer=masked_scorer,
                                device=device,
                            ),
                            validation_interval=cfg.learner.validation_interval,
                            metrics={"nDCG@10": True},
                        )

                        checkpoints = MainListener.C(
                            id="sparsity_logger",
                            interval=cfg.learner.checkpoint_listener_interval,
                            sparsity_interval=cfg.learner.sparsity_listener_interval,
                        )
    
                        learner = Learner.C(
                            # Misc settings
                            device=device,
                            random=random,
                            # How to train the model
                            trainer=monobert_trainer,
                            # The model to train
                            model=masked_scorer,
                            # Optimization settings
                            steps_per_epoch=cfg.learner.optimization.steps_per_epoch,
                            optimizers=optimizer,
                            max_epochs=cfg.learner.optimization.max_epochs,
                            checkpoint_interval=cfg.learner.checkpoint_interval,
                            # The listeners (here, for validation)
                            listeners=[validation, checkpoints],
                            # The hook used for evaluation
                            hooks=[setmeta(DistributedHook.C(models=[masked_scorer]), True)],
                        )

                        # Submit job and link
                        outputs.append(learner.submit(launcher=optimization_launcher, init_tasks=[]))
                        tensorboard_service.add(learner, learner.logpath)

    return outputs

@ir_experiment()
def run(
    helper: IRExperimentHelper, cfg: MaskLearning
) -> PaperResults:
    cfg.check_end_layer()
    
    advanced_ablation_optimization(
        cfg=cfg,
        ranker_id=cfg.ranker_id,
        base_hf_id=cfg.base_hf_id,
        device=cfg.device,
        random=cfg.random,
        tensorboard_service=helper.tensorboard_service
    )