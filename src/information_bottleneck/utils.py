import re
import sys
import torch
import torch.nn.functional as F
import json
import os
from enum import Enum
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import ir_measures
from ir_measures import *
from scipy import stats

from xpmir.text import TokenizedTexts
from xpmir.learning.context import TrainerContext, Loss
from xpmir.distributed import DistributableModel
from xpmir.rankers import LearnableScorer, ScorerOutputType
from xpmir.learning.learner import LearnerListener, Learner, LearnerListenerStatus
from xpmir.learning.context import TrainState, TrainerContext
from xpmir.letor.trainers.pairwise import PairwiseRecords, PairwiseTrainer
from xpmir.letor.distillation.pairwise import DistillationPairwiseLoss
from xpmir.learning.metrics import ScalarMetric
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from experimaestro import Task, Param, Config, Meta, Constant, pathgenerator, Annotated
from datamaestro import prepare_dataset
from datamaestro_text.data.ir import TextItem, IDItem, PairwiseSampleDataset

from utils import untuple, INPUT_PART_TO_POSITION, get_relevance_levels
from ablations.utils import AblationOutput

import logging

logging.basicConfig(level=logging.INFO)    

def threshold_mask(mask, threshold):
    mask = mask.clone()
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    return mask

class MaskingStrategyEnum(Enum):
    SIGMOID = "sigmoid"
    GUMBEL_SOFTMAX = "gumbel_softmax"
    GUMBEL_SIGMOID = "gumbel_sigmoid"

    @staticmethod
    def values():
        return [e.value for e in MaskingStrategyEnum]

    def __str__(self):
        return self.value
    
class MaskingStrategy(Config, torch.nn.Module):
    __xpmid__="src.ablations.mask_utils.maskingstrategy"

    num_categories: Param[int]

    num_attention_heads: Param[int]

    start_layer: Param[int]

    end_layer: Param[int]

    max_seq_len: Param[int]

    masks_initial_value: Param[float]

    def initialize(self):
        super().__init__()
        self.padding_idx = self.num_categories**2

        self.n_dim = None
        self.masks_weights = None
        self.current_masks = None
        self.temperature = 1.0

   # @torch.compile
    def init_current_masks(self, inputs, device):
        # Build masks through differentiable ops only so gradients can flow to mask embeddings.
        input_indices = inputs.int()
        layer_masks = []
        for layer_nb in range(self.start_layer, self.end_layer):
            head_masks = []
            for attention_head in range(self.num_attention_heads):
                embedding = self.masks_weights[layer_nb * self.num_attention_heads + attention_head]
                head_masks.append(embedding(input_indices))
            layer_masks.append(torch.stack(head_masks, dim=1))
        self.current_masks = torch.stack(layer_masks, dim=0)

    def reset_masks(self):
        assert self.current_masks is not None, "Masks are already set to None"
        del self.current_masks
        self.current_masks = None

    def load_masks_weights(self, new_masks_weights, device, requires_grad: bool):
        for idx, mask in enumerate(self.masks_weights):
            mask.weight = torch.nn.Parameter(new_masks_weights[f"masking_strategy.masks_weights.{idx}.weight"].to(device))
            mask.weight.requires_grad = requires_grad

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def __call__(self):
        raise NotImplementedError()
    
class SigmoidMaskingStrategy(MaskingStrategy):
    __xpmid__="src.ablations.mask_utils.sigmoidmaskingstrategy"

    def initialize(self):
        super().initialize()
        # This is what we'll actually be learning. These embeddings in fact correspond to the weights applied to each block of the attention matrix.
        # There are one embedding table per attention head per layer as we want each one of them to be independant 
        # Each block corresponds to a pair of input parts (e.g. block of positions where `query attends to document`, etc.)
        self.n_dim = 1
        self.masks_weights = torch.nn.ParameterList([torch.nn.Embedding(self.num_categories**2+1, self.n_dim, padding_idx=self.num_categories**2) for _ in range((self.end_layer - self.start_layer) * self.num_attention_heads)])
        for mask in self.masks_weights:
            torch.nn.init.constant_(mask.weight[:-1], self.masks_initial_value) # We initialise every weight (except for the padding tokens but they are not used) to account for the Sigmoid transformation

    def __call__(self, layer_nb: int, **kwargs):
        mask = torch.nn.functional.sigmoid(self.current_masks[layer_nb])
        return torch.select(mask, -1, 0)
    
class GumbelSoftmaxMaskingStrategy(MaskingStrategy):
    __xpmid__="src.ablations.mask_utils.gumbelsoftmaxmaskingstrategy"

    tau: Param[float]

    def initialize(self):
        super().initialize()
        # This is what we'll actually be learning. These embeddings in fact correspond to the weights applied to each block of the attention matrix.
        # There are one embedding table per attention head per layer as we want each one of them to be independant 
        # Each block corresponds to a pair of input parts (e.g. block of positions where `query attends to document`, etc.)
        self.n_dim = 2
        self.masks_weights = torch.nn.ParameterList([torch.nn.Embedding(self.num_categories**2+1, self.n_dim, padding_idx=self.num_categories**2) for _ in range((self.end_layer - self.start_layer) * self.num_attention_heads)])
        for mask in self.masks_weights:
            torch.nn.init.constant_(mask.weight[:-1][:,0], self.masks_initial_value)
            torch.nn.init.constant_(mask.weight[:-1][:,1], self.masks_initial_value)
    
    def __call__(self, layer_nb: int, threshold: Optional[float] = None, testing: bool = False):
        if not testing: # If we are training, we use the Gumbel Softmax
            mask = torch.nn.functional.gumbel_softmax(self.current_masks[layer_nb], tau=self.tau, hard=True)
            mask = torch.select(mask, -1, 0)
        else:
            mask = torch.nn.functional.softmax(self.current_masks[layer_nb], dim=-1) # If not training, we don't the Gumbel noise, just the softmax
            assert threshold is not None, "Threshold must be provided when not training"
            mask = (torch.select(mask, -1, 0) > threshold).float()
        return mask
    
class GumbelSigmoidMaskingStrategy(MaskingStrategy):
    __xpmid__="src.ablations.mask_utils.gumbelsigmoidmaskingstrategy"

    tau: Param[float]

    def initialize(self):
        super().initialize()
        # This is what we'll actually be learning. These embeddings in fact correspond to the weights applied to each block of the attention matrix.
        # There are one embedding table per attention head per layer as we want each one of them to be independant 
        # Each block corresponds to a pair of input parts (e.g. block of positions where `query attends to document`, etc.)
        self.n_dim = 1
        self.masks_weights = torch.nn.ParameterList([torch.nn.Embedding(self.num_categories**2+1, self.n_dim, padding_idx=self.num_categories**2) for _ in range((self.end_layer - self.start_layer) * self.num_attention_heads)])
        for mask in self.masks_weights:
            torch.nn.init.constant_(mask.weight[:-1], self.masks_initial_value) # We initialise every weight at 3.0 (except for the padding tokens but they are not used) to account for the Sigmoid transformation

    def gumbel_sigmoid(self, input):
        U1 = torch.Tensor.uniform_(torch.zeros_like(input))
        U2 = torch.Tensor.uniform_(torch.zeros_like(input))
        s = torch.nn.functional.sigmoid(input - torch.log(torch.log(U1)/torch.log(U2)) / self.tau)
        with torch.no_grad():
            indicator = torch.where(s > 0.5, 1, 0)
            straight_through = indicator - s
        mask = straight_through + s
        return mask

    def init_current_masks(self, inputs, device):
        self.current_masks = torch.zeros((self.end_layer - self.start_layer, len(inputs), self.num_attention_heads, self.max_seq_len, self.max_seq_len, self.n_dim), device=device)
        self.current_masks.requires_grad = False
        # Now we need to convert each type of combination to the corresponding weight in a learned embedding table
        for layer_nb in range(self.start_layer, self.end_layer):
            for attention_head in range(self.num_attention_heads):
                converted_embedding_weights = self.gumbel_sigmoid(self.masks_weights[layer_nb * self.num_attention_heads + attention_head].weight[:-1])
                converted_embedding_weights = torch.concatenate((converted_embedding_weights, self.masks_weights[layer_nb * self.num_attention_heads + attention_head].weight[-1].unsqueeze(dim=1)),dim=0)
                self.current_masks[layer_nb, :, attention_head] = torch.nn.functional.embedding(inputs.int(), converted_embedding_weights)

    def __call__(self, layer_nb: int, **kwargs):
        return torch.select(self.current_masks[layer_nb], -1, 0)

class TestingMaskForCrossScorer(torch.nn.Module):
    # Hook used to mask the output of some layer given a pruning scheme
    def __init__(self, model: torch.nn.Module, start_layer: int, end_layer: int, ranker_id: str, base_model_id: str, max_seq_len: int, device, masking_strategy: MaskingStrategy, mask_path: Path, threshold: float):
        super().__init__()
        self.model = model
        self.hooks_handles = list()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.ranker_id = ranker_id
        self.base_model_id = base_model_id
        self.config = BertConfig.from_pretrained(self.ranker_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.max_seq_len = max_seq_len
        self.device = device
        self.masking_strategy = masking_strategy
        self.masking_strategy.initialize()
        self.masking_strategy.load_masks_weights(torch.load(mask_path), self.device, requires_grad=False)
        self.threshold = threshold # Threshold above which masks are set to 1, below to 0

        for layer_nb in range(self.start_layer, self.end_layer):
            layer = dict([*self.model.named_modules()])[f"bert.encoder.layer.{layer_nb}.attention.self.dropout"]
            self.hooks_handles.append(layer.register_forward_hook(
                self.prune_dropout(layer_nb))
            )
    
    def prune_dropout(self, layer_nb) -> Callable:
        def hook(model, input, output):
            output = untuple(output)
            mask = self.masking_strategy(layer_nb, self.threshold, testing=True)
            masked_output = output * mask.to(self.device)
            return masked_output
        return hook
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()

    def build_lookup_matrix(self, input_ids):
        A = torch.where(input_ids == self.tokenizer.sep_token_id, 1.0, 0.0)
        A += torch.where(input_ids == self.tokenizer.cls_token_id, 1.0, 0.0)
        A += torch.where(input_ids == self.tokenizer.pad_token_id, -torch.inf, 0.0)

        # Add a column at the beginning of the matrix in prevision of the `cumsum`
        A = torch.hstack([torch.zeros(A.shape[0], 1).to(A.device), A])
        A = A[:, 1:] + A[:, :-1]
        A = A.cumsum(dim=1) - 1 # -1 so that CLS is 0, query 1, etc.
        A = A.unsqueeze(dim=1) + (A*self.masking_strategy.num_categories).unsqueeze(dim=2) # 5 different types of input parts, so 25 combinations in total
        # `.unsqueeze()` is here so that final shape of A is: `batch_size` x `max_seq_len` x `max_seq_len`
        A = torch.where(A == -torch.inf, self.masking_strategy.num_categories**2, A)
        return A

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):        
        parsed_inputs = self.build_lookup_matrix(input_ids) # parsed inputs is the conversion of the input_ids to the corresponding combination of input parts
        self.masking_strategy.init_current_masks(parsed_inputs, self.device)
                                    
        with torch.no_grad():
            kwargs = {}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids.to(self.device)

            model_outputs = self.model(
                input_ids, attention_mask=attention_mask.to(self.device), **kwargs
            )
    
        # In the forward also returns the masks / sizes to re-weight the loss
        self.masking_strategy.reset_masks()
        return model_outputs.logits

class LearningMaskForCrossScorer(LearnableScorer, DistributableModel):
    __xpmid__="src.ablations.mask_utils.learningmaskforcrossscorer"

    ranker_id: Param[str]

    base_model_id: Param[str]

    masking_strategy: Param[MaskingStrategy]

    start_layer: Param[int]

    end_layer: Param[int]

    max_seq_len: Param[int]

    # Here for legacy reasons, in case people would like to use the PairWiseTrainer without Distillation
    outputType: Param[ScorerOutputType] = ScorerOutputType.PROBABILITY 

    version: Constant[int] = 2

    def __initialize__(self, options):
        super().__initialize__(options)

        self.masking_strategy.initialize()

        self._dummy_params = torch.nn.Parameter(torch.Tensor())

        self.model = BertForSequenceClassification.from_pretrained(self.ranker_id)
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

        self.is_teacher = False

        if self.model.config.num_labels == 1:
            self.activation_fct = lambda x, dim: F.sigmoid(x)
            self.target_position = 0
        else:
            self.activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            self.target_position = 1

        self.hooks_handles = list()

        # Register the hooks for the dropout layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Dropout) and any(f"bert.encoder.layer.{i}." in name for i in range(self.start_layer, self.end_layer)):
                layer_nb = int(re.search(r"bert\.encoder\.layer\.(\d+)\.", name).group(1)) # Extracts the layer number from the name
                self.hooks_handles.append(module.register_forward_hook(self.prune_dropout(name, layer_nb)))

        self.hooks_handles.append(self.model.dropout.register_forward_hook(self.prune_dropout("dropout", self.end_layer)))
        self.hooks_handles.append(self.model.bert.embeddings.dropout.register_forward_hook(self.prune_dropout("bert.embeddings.dropout", self.start_layer)))  # The dropout layer at the end of the model
        # For modules matching this pattern, we want to apply the masking IF and only if we are not in the teacher
        self.pattern = re.compile(r"^bert\.encoder\.layer\.(\d+)\.attention\.self\.dropout$")
            

    @property
    def device(self):
        return self._dummy_params.device
    
    @property
    def masks_weights(self):
        return self.masking_strategy.masks_weights
    
    def prune_dropout(self, name: str, layer_nb: int) -> Callable:
        """ This hook aims at skipping all dropout modules in the model and only in the case where we are in the student and the dropout is in the self.attention,\\
             apply the masking strategy"""
        def hook(model, input, output):
            if self.is_teacher:
                return untuple(input)
            elif not self.pattern.fullmatch(name):
                return untuple(input)
            else:
                assert self.masking_strategy.current_masks is not None, "Attention scores mask must be set before calling forward"
                mask = self.masking_strategy(layer_nb, testing=False)
                input = untuple(input)
                masked_input = input * mask.to(self.device)
                return masked_input
        return hook
    
    def build_lookup_matrix(self, input_ids):
        A = torch.where(input_ids == self.tokenizer.sep_token_id, 1.0, 0.0)
        A += torch.where(input_ids == self.tokenizer.cls_token_id, 1.0, 0.0)
        A += torch.where(input_ids == self.tokenizer.pad_token_id, -torch.inf, 0.0)

        # Add a column at the beginning of the matrix in prevision of the `cumsum`
        A = torch.hstack([torch.zeros(A.shape[0], 1).to(A.device), A])
        A = A[:, 1:] + A[:, :-1]
        A = A.cumsum(dim=1) - 1 # -1 so that CLS is 0, query 1, etc.
        A = A.unsqueeze(dim=1) + (A*self.masking_strategy.num_categories).unsqueeze(dim=2) # 5 different types of input parts, so 25 combinations in total
        # `.unsqueeze()` is here so that final shape of A is: `batch_size` x `max_seq_len` x `max_seq_len`
        A = torch.where(A == -torch.inf, self.masking_strategy.num_categories**2, A)
        return A
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()

    def batch_tokenize(self,
        texts: Union[List[str], List[Tuple[str, str]]],
        batch_first=True,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        else:
            maxlen = min(maxlen, self.tokenizer.model_max_length)

        assert batch_first, "Batch first is the only option"

        r = self.tokenizer(
            list(texts),
            max_length=maxlen,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_length=True,
            return_attention_mask=mask,
        )
        return TokenizedTexts(
            None,
            r["input_ids"].to(self.device),
            r["length"],
            r.get("attention_mask", None),
            r.get("token_type_ids", None),  # if r["token_type_ids"] else None
        )

    def forward(
        self, 
        inputs: List[Tuple],
        temperature: float = 1.0,
        is_teacher: bool = False,
        is_input_tokenized: bool = False,
        info: TrainerContext = None
    ):  
        self.is_teacher = is_teacher
        if not is_input_tokenized:
            tokenized_inputs = self.batch_tokenize(
                [
                    (tr[TextItem].text, dr[TextItem].text)
                    for tr, dr in zip(inputs.topics, inputs.documents)
                ], 
                maxlen=self.max_seq_len, 
                mask=True
            )
        else:
            tokenized_inputs = inputs
        
        if temperature is not None:
            # During validation, do not update the temperature
            self.masking_strategy.set_temperature(temperature)
            
        if not is_teacher:
            parsed_inputs = self.build_lookup_matrix(tokenized_inputs.ids) # parsed inputs is the conversion of the input_ids to the corresponding combination of input parts
            self.masking_strategy.init_current_masks(parsed_inputs, self.device)
    
        with torch.set_grad_enabled(torch.is_grad_enabled() and not is_teacher): # We want to backprop on the mask so grad is enabled
            kwargs = {}
            if tokenized_inputs.token_type_ids is not None:
                kwargs["token_type_ids"] = tokenized_inputs.token_type_ids.to(self.device)

            model_outputs = self.model(
                tokenized_inputs.ids, attention_mask=tokenized_inputs.mask.to(self.device), **kwargs
            )
            probas = self.activation_fct(model_outputs.logits, dim=1)
    
        # In the forward also returns the masks / sizes to re-weight the loss
        if not is_teacher:
            self.masking_strategy.reset_masks()
        return untuple(probas[:,self.target_position],)
        
    def distribute_models(self, update):
        self.model = update(self.model)

class DistillationPairwiseTrainerWithSparsification(PairwiseTrainer):
    """Pairwise trainer uses samples of the form (query, positive, negative)"""
    __xpmid__="src.ablations.mask_utils.distillationpairwisetrainerwithsparsification"

    alpha: Param[float] = 0.01
    lossfn: Param[DistillationPairwiseLoss]

    version: Constant[int] = 2

    def train_batch(self, records: PairwiseRecords):
        # Get the next batch and compute the scores for each query/document
        if self.ranker.training:
            self.ranker.eval()
        rel_scores = self.ranker(records, info=self.context)
        with torch.no_grad():
            # pass the ranker in evaluation mode to avoid dropout for teacher scores
            teacher_scores = self.ranker(records, is_teacher=True, info=self.context).reshape(2, len(records)).T
        # rel_scores = untuple(rel_scores)
        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        self.lossfn.process(pairwise_scores, teacher_scores, self.context)

        regularization_loss = torch.mean(torch.cat([torch.mean(mask.weight[:-1].sigmoid(), 0, keepdim=True) for mask in self.ranker.masks_weights]))

        # Warning: we remove the last dimension because it corresponds to padding and is fixed, thus potentially hurtful to the learning process
        self.context.add_loss(Loss(f"masks-regularization", regularization_loss, self.alpha))        

        self.context.add_metric(
            ScalarMetric(
                "accuracy", float(self.acc(pairwise_scores).item()), len(rel_scores)
            )
        )

class MainListener(LearnerListener):
    """Learning validation early-stopping

    Every `interval` epochs, the listener will save the model and the optimizer.
    It also controls when to log the histogram of the distribution of the weights.
    """
    __xpmid__="src.ablations.mask_utils.mainlistener"

    path: Annotated[Path, pathgenerator("checkpoints")]
    """Path to the checkpoints"""

    interval: Param[int] = 10
    """Epochs between each checkpoint"""

    sparsity_interval: Param[int] = 10

    def __validate__(self):
        assert (
            self.interval > 0
        ), "Interval should be superior to 0"

    def initialize(self, learner: Learner, context: TrainerContext):
        super().initialize(learner, context)

        self.path.mkdir(exist_ok=True, parents=True)
        (self.path / self.id).mkdir(exist_ok=True, parents=True)

    def __call__(self, state: TrainState):
        if state.epoch % self.interval == 0:
            logging.info(
                f"Saving the checkpoint {state.epoch}"
            )
            self.context.copy(self.path / f"{self.id}/checkpoint-{state.epoch}.pt")
            # Diagnostic check: verify mask parameters actually changed
            if not hasattr(self, '_prev_mask_params'):
                self._prev_mask_params = [mask.weight.data.clone().detach().cpu() for mask in state.model.masks_weights]
            else:
                param_changed = False
                for idx, (mask, prev) in enumerate(zip(state.model.masks_weights, self._prev_mask_params)):
                    delta = (mask.weight.data.clone().detach().cpu() - prev).abs().max().item()
                    if delta > 1e-10:
                        param_changed = True
                        break
                if not param_changed:
                    logging.warning(
                        f"Mask parameters have not changed since last checkpoint (epoch {state.epoch - self.interval}). "
                        f"Check if optimizer is stepping masks, or if learning rate is too low (current default lr=1.0)."
                    )
                self._prev_mask_params = [mask.weight.data.clone().detach().cpu() for mask in state.model.masks_weights]
        
        if state.epoch % self.sparsity_interval == 0:
            logging.info(
                f"Reporting the sparsity levels at epoch {state.epoch}"
            )
            copy_masks = [mask.weight.data.clone().detach().cpu() for mask in state.model.masks_weights]
            if copy_masks[0].shape[-1] == 2:
                self.context.writer.add_histogram('masks-sparsity', np.array([torch.select(torch.nn.functional.gumbel_softmax(mask, tau=1.0, hard=True), -1, 0)[:-1] for mask in copy_masks]), state.epoch)
            else:
                self.context.writer.add_histogram('masks-sparsity', np.array([torch.nn.functional.sigmoid(mask[:-1]) for mask in copy_masks]), state.epoch)
            # Warning: we remove the last dimension because it corresponds to padding and is fixed, thus potentially hurtful to the learning process

        return LearnerListenerStatus.DONT_STOP

class AdvancedAblationTestFromPath(Task):
    masking_strategy: Param[str]

    ranker_id: Param[str] 

    base_hf_id: Param[str]

    dataset_name: Param[str]

    parsed_dataset: Param[PairwiseSampleDataset]

    start_layer: Param[int]

    end_layer: Param[int]

    max_seq_len: Param[int]

    mask_path: Param[str]

    threshold: Param[float]

    device: Meta[Device] = DEFAULT_DEVICE

    output_path: Annotated[Path, pathgenerator("output")]

    def task_outputs(self, dep: Callable[[Config], None]) -> AblationOutput:
        return dep(AblationOutput.C(task=self))

    def execute(self):
        parsed_dataset_with_qrels = get_relevance_levels(self.dataset_name, self.parsed_dataset)
        self.device.execute(self.device_execute, parsed_dataset_with_qrels)

    def device_execute(self, device_information: DeviceInformation, parsed_dataset_with_qrels: Dict[str, Dict[str, float]]):
        passages = prepare_dataset(self.dataset_name)
        topics = prepare_dataset(self.dataset_name + '.queries')
    
        ranker_config = BertConfig.from_pretrained(self.ranker_id)
        original_model = BertForSequenceClassification.from_pretrained(self.ranker_id)
        model = BertForSequenceClassification.from_pretrained(self.ranker_id)

        if model.config.num_labels == 1:
            activation_fct = lambda x, dim: F.sigmoid(x)
            target_position = 0
        else:
            activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            target_position = 1

        logging.info(f"Starting the ablations for the dataset {self.dataset_name}.")
        proba_diff = 0
        num_samples = 0
        # Check for already processed (query_id, pid) pairs
        processed_pairs = set()
        log_file = f"{self.output_path}/ablation_logs.jsonl"
        # If logs exist, load them and accumulate processed pairs and proba_diff
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        processed_pairs.add((entry["query_id"], entry["passage_id"]))
                        proba_diff += abs(entry["original_proba"] - entry["ablation_proba"])
                        num_samples += 1
                logging.info(f"Loaded {num_samples} existing logs from {log_file}.")
            except Exception as e:
                logging.warning(f"Could not load existing logs from {log_file}: {e}")
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if self.masking_strategy == str(MaskingStrategyEnum.SIGMOID):
            masking_strategy = SigmoidMaskingStrategy(
                num_categories=len(INPUT_PART_TO_POSITION.keys()), 
                num_attention_heads=ranker_config.num_attention_heads, 
                start_layer=self.start_layer, 
                end_layer=self.end_layer, 
                max_seq_len=self.max_seq_len, 
                masks_initial_value=0.0
            )
        elif self.masking_strategy ==  str(MaskingStrategyEnum.GUMBEL_SOFTMAX):
            masking_strategy = GumbelSoftmaxMaskingStrategy(
                num_categories=len(INPUT_PART_TO_POSITION.keys()), 
                num_attention_heads=ranker_config.num_attention_heads, 
                start_layer=self.start_layer, 
                end_layer=self.end_layer, 
                max_seq_len=self.max_seq_len, 
                masks_initial_value=0.0,
                tau=1.0
            )
        elif self.masking_strategy == str(MaskingStrategyEnum.GUMBEL_SIGMOID):
            masking_strategy = GumbelSigmoidMaskingStrategy(
                num_categories=len(INPUT_PART_TO_POSITION.keys()), 
                num_attention_heads=ranker_config.num_attention_heads, 
                start_layer=self.start_layer, 
                end_layer=self.end_layer, 
                max_seq_len=self.max_seq_len, 
                masks_initial_value=self.masks_initial_value,
                tau=1.0
            )
        else:
            raise ValueError("Masking strategy not recognized.")

        ablation_model = TestingMaskForCrossScorer(
            model=model,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            ranker_id=self.ranker_id,
            base_model_id=self.base_hf_id,
            max_seq_len=self.max_seq_len,
            device=device_information.device,
            masking_strategy=masking_strategy,
            mask_path=Path(self.mask_path),
            threshold=self.threshold
        )

        original_model.eval()
        ablation_model.model.eval()

        original_model.to(device_information.device)
        ablation_model.model.to(device_information.device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_hf_id)

        for query_id, pid_to_rel in tqdm(parsed_dataset_with_qrels.items()):
            query = topics.topic_ext(query_id)
            for pid, rel in tqdm(pid_to_rel.items()):
                if (query_id, pid) in processed_pairs:
                    continue  # Skip already processed pairs
                passage = passages.documents.document_ext(pid)
                inputs = tokenizer(
                    query[TextItem].text, 
                    passage[TextItem].text,
                    max_length=self.max_seq_len, # True max length is 512 but it gets too long to compute
                    truncation=True,
                    padding="max_length", # To get every sample to the same size
                    return_attention_mask=True,
                    return_tensors="pt"
                ).to(device_information.device)
                inputs.requires_grad = False

                with torch.no_grad():
                    original_logits = original_model(**inputs).logits.detach().cpu()
                    original_proba = activation_fct(original_logits, dim=1)[0][target_position].detach().cpu().numpy()
                    ablated_logits = ablation_model(**inputs).detach().cpu() 
                    ablated_proba = activation_fct(ablated_logits, dim=1)[0][target_position].detach().cpu().numpy()
                log_entry = {
                        "query_id": query[IDItem].id,
                        "passage_id": passage[IDItem].id,
                        "passage_relevance": rel,
                        "original_logits": original_logits.numpy().tolist(),
                        "original_proba": float(original_proba),
                        "ablation_logits": ablated_logits.numpy().tolist(),
                        "ablation_proba": float(ablated_proba)
                    }
                proba_diff += abs(original_proba - ablated_proba)
                num_samples += 1

                # Save log after each entry for robustness (append as JSONL)
                with open(f"{self.output_path}/ablation_logs.jsonl", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
        
        logging.info(f"Average difference in probabilities: {proba_diff/num_samples}")

        if not self.output_path.exists():
            os.mkdir(self.output_path)
        value_file = f"{self.output_path}/averaged_diff.npy"
        np.save(value_file, proba_diff/num_samples)
        logging.info(f"Logs and average difference saved to {self.output_path}.")

class AgregationAdvancedAblationTests(Task):
    advanced_ablation_outputs: Param[List[AblationOutput]]

    storage_path: Annotated[Path, pathgenerator("storage")]

    def execute(self):
        original_results_per_query = dict()
        ablated_results_per_query = dict()
        qrels = dict()

        for ablation_output in self.advanced_ablation_outputs:
            ablation_results = np.load(f"{ablation_output.task.output_path}/averaged_diff.npy")
            dataset = ablation_output.task.dataset_name
            original_results_per_query[dataset] = dict()
            ablated_results_per_query[dataset] = dict()
            qrels[dataset] = dict()
            if Path(f"{ablation_output.task.output_path}/ablation_logs.jsonl").exists():
                logging.info(f"Processing logs from {ablation_output.task.output_path}/ablation_logs.jsonl for dataset {dataset}")
                with open(f"{ablation_output.task.output_path}/ablation_logs.jsonl", "r") as f:
                    for line in f:
                        dico = json.loads(line)
                        if dico["query_id"] not in ablated_results_per_query[dataset].keys():
                            ablated_results_per_query[dataset][dico["query_id"]] = dict()
                        ablated_results_per_query[dataset][dico["query_id"]][dico["passage_id"]] = float(dico["ablation_proba"])

                        if dico["query_id"] not in original_results_per_query[dataset].keys():
                            original_results_per_query[dataset][dico["query_id"]] = dict()
                            qrels[dataset][dico["query_id"]] = dict()
                        original_results_per_query[dataset][dico["query_id"]][dico["passage_id"]] = float(dico["original_proba"])
                        qrels[dataset][dico["query_id"]][dico["passage_id"]] = dico["passage_relevance"]

            elif Path(f"{ablation_output.task.output_path}/ablation_logs.npy").exists(): 
                logging.info(f"Processing logs from {ablation_output.task.output_path}/ablation_logs.npy for dataset {dataset}")
                activations = np.load(f"{ablation_output.task.output_path}/ablation_logs.npy", allow_pickle=True)
                for dico in activations:
                    if dico["query_id"] not in ablated_results_per_query[dataset].keys():
                        ablated_results_per_query[dataset][dico["query_id"]] = dict()
                    ablated_results_per_query[dataset][dico["query_id"]][dico["passage_id"]] = float(dico["ablation_proba"])
                    if dico["query_id"] not in original_results_per_query[dataset].keys():
                        original_results_per_query[dataset][dico["query_id"]] = dict()
                        qrels[dataset][dico["query_id"]] = dict()
                    original_results_per_query[dataset][dico["query_id"]][dico["passage_id"]] = float(dico["original_proba"])
                    qrels[dataset][dico["query_id"]][dico["passage_id"]] = dico["passage_relevance"]
            else:
                raise ValueError(f"No logs found in {ablation_output.task.output_path}.")
         
        base_run_metrics_per_dataset_per_qid = dict()
        base_run_metrics_per_dataset = dict()
        ablated_run_metrics_per_dataset_per_qid = dict()
        ablated_run_metrics_per_dataset = dict()
        ndcg_p_value_per_dataset = dict()
        for dataset in ablated_results_per_query.keys():
            logging.info(f"Computing nDCG values for {dataset}.")
            queries = list(qrels[dataset].keys())
            documents = list({doc for query in qrels[dataset] for doc in qrels[dataset][query]})

            qrels_scores = np.array([
                [qrels[dataset][query].get(doc, 0) for doc in documents]
                for query in queries
            ])
            runs = np.array([
                [ablated_results_per_query[dataset][query].get(doc, 0) for doc in documents]
                for query in queries
            ])
            # Compute NDCG scores for each query
            base_run_metrics_per_dataset_per_qid[dataset] = {"ndcg": {m.query_id: m.value for m in ir_measures.iter_calc([nDCG@10], qrels[dataset], original_results_per_query[dataset])}}
            base_run_metrics_per_dataset[dataset] = ir_measures.calc_aggregate([nDCG@10], qrels[dataset], original_results_per_query[dataset])
            ndcg_scores = []
            for qrel_score, single_run in zip(qrels_scores, runs):
                ndcg_scores.append(ndcg_score([qrel_score], [single_run], k=10, ignore_ties=False))

            ablated_run_metrics_per_dataset_per_qid[dataset] = {"ndcg": {query: score for query, score in zip(queries, ndcg_scores)}}
            ablated_run_metrics_per_dataset[dataset] = ndcg_score(qrels_scores, runs, k=10, ignore_ties=False)

            ndcg_t_statistic, ndcg_p_value = stats.ttest_ind(
                [base_run_metrics_per_dataset_per_qid[dataset]["ndcg"][v] for v in qrels[dataset].keys()], 
                [ablated_run_metrics_per_dataset_per_qid[dataset]["ndcg"][v] for v in qrels[dataset].keys()]
            )
            ndcg_p_value_per_dataset[dataset] = ndcg_p_value

        if not self.storage_path.exists():
            os.mkdir(self.storage_path)
        
        pd.DataFrame.from_dict(data=base_run_metrics_per_dataset, orient='index').to_csv(f"{self.storage_path}/base_run_metrics.csv", header=True)
        pd.DataFrame.from_dict(data=ablated_run_metrics_per_dataset, orient='index').to_csv(f"{self.storage_path}/ablated_run_metrics.csv", header=True)
        pd.DataFrame.from_dict(data=ndcg_p_value_per_dataset, orient='index').to_csv(f"{self.storage_path}/ndcg_p_values.csv", header=True)
        np.save(f"{self.storage_path}/ablation_results.npy", ablation_results)
