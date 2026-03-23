import gc
import glob
import gzip
import json
import os
from pathlib import Path
import pickle
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datamaestro_text.data.ir import TextItem
from xpmir.learning.devices import DeviceInformation
from xpmir.letor.records import PairwiseRecordWithTarget
from xpmir.rankers import Scorer
from xpmir.utils.iter import SkippingIterator
from xpmir.learning import Random, ModuleInitMode
from xpmir.letor.samplers import Sampler
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from experimaestro import Task, Param, Meta, Annotated
from experimaestro.generators import pathgenerator

from extractors import OutputsExtractorInfoNCE
from nig.baseline import Baseline
from nig.integrated_gradients import (
    getAlphaParameters,
    getSlopes_infonce,
    _get_scaled_inputs,
    _get_ig_error
)


import logging

from utils import get_interesting_modules, get_token_types_spans

logging.basicConfig(level=logging.INFO)

def nig_infonce_full_model(
    cross_scorer, 
    pos_input_embeddings,
    pos_token_type_ids,
    pos_attention_mask,
    pos_baseline_embeddings, 
    neg_input_embeddings,
    neg_token_type_ids,
    neg_attention_mask,
    neg_baseline_embeddings,
    num_labels: int,
    batch_size: int, 
    steps: int,
    compute_error: bool = False,
    adaptive_sampling: bool = False
) -> Dict:
    """
    Compute the attribution (Neuron Integrated Gradients) of each unit for all the interesting modules in the model.

    :param torch.nn.Module model: Model for which to compute the conductance.
    :param torch.Tensor input_embeddings: Embeddings of the input.
    :param torch.Tensor token_type_ids: Token type ids of the input.
    :param torch.Tensor attention_mask: Attention mask on the input.
    :param torch.Tensor baseline_embeddings: Embedding of the baseline for the given input.
    :param int pred_label: Label predicted by the base model for the given
    :param int num_reps: Number of iteration to approximate the integrated gradients.
    :param int batch_size: Batch size used for each iteration (true number of steps is batch_size x num_reps).
    :return Dict: Attribution for each activation unit for each layer in the model.
    """
    layer_names = get_interesting_modules(
        model=cross_scorer.model,
    )
    product_storage = dict() 
    all_diffs = list()

    # Get the adaptive steps and generate the embeddings of each input along the path at every step
    pos_baseline_diff=torch.sub(pos_input_embeddings, pos_baseline_embeddings).detach()
    neg_baseline_diff=torch.sub(neg_input_embeddings, neg_baseline_embeddings).detach()
    if adaptive_sampling:
        slopes, step_size = getSlopes_infonce(
            steps=steps,
            batch_size=batch_size, 
            model=cross_scorer.model,
            pos_token_type_ids=pos_token_type_ids,
            pos_attention_mask=pos_attention_mask,
            pos_baseline_embeddings=pos_baseline_embeddings.detach(),
            neg_token_type_ids=neg_token_type_ids,
            neg_attention_mask=neg_attention_mask,
            neg_baseline_embeddings=neg_baseline_embeddings.detach(),
            pos_baseline_diff=pos_baseline_diff,
            neg_baseline_diff=neg_baseline_diff,
            device=cross_scorer.device
        )
        _, alpha_substep_size = getAlphaParameters(slopes, steps, step_size)

    else:
        alpha_substep_size = [float(i)/ (steps-1) for i in range(steps)]

    list_scaled_pos_embeddings = _get_scaled_inputs(
        baseline_tensor=pos_baseline_embeddings[0].detach().cpu().numpy(), 
        baseline_diff=pos_baseline_diff[0].detach().cpu().numpy(),
        alpha_substep_size=alpha_substep_size,
        steps=steps,
        batch_size=batch_size,
        device=cross_scorer.device
    )
    list_scaled_neg_embeddings = _get_scaled_inputs(
        baseline_tensor=neg_baseline_embeddings[0].detach().cpu().numpy(), 
        baseline_diff=neg_baseline_diff[0].detach().cpu().numpy(),
        alpha_substep_size=alpha_substep_size,
        steps=steps,
        batch_size=batch_size,
        device=cross_scorer.device
    )

    extractor = OutputsExtractorInfoNCE(
        model=cross_scorer.model,
        layer_names=layer_names
    )

    # Use these embeddings to compute the IG
    for i in tqdm(range(len(list_scaled_pos_embeddings))):
        # First get the different outputs for the positives
        batch_pos_inputs = torch.Tensor(list_scaled_pos_embeddings[i]).to(torch.float)
        batch_pos_inputs.requires_grad = True
        pos_outputs = extractor.pos_forward(batch_pos_inputs, token_type_ids=pos_token_type_ids, attention_mask=pos_attention_mask)

        # then for the negatives
        batch_neg_inputs = torch.Tensor(list_scaled_neg_embeddings[i]).to(torch.float)
        batch_neg_inputs.requires_grad = True
        neg_outputs = extractor.neg_forward(batch_neg_inputs, token_type_ids=neg_token_type_ids, attention_mask=neg_attention_mask)
       
        if num_labels== 1:
            # pos_outputs = cross_scorer.model.classifier(pos_outputs.logits)
            # neg_outputs = cross_scorer.model.classifier(neg_outputs.logits)
            logit_pos = pos_outputs.logits
            logit_neg = neg_outputs.logits

            logit_pos_d = logit_pos.detach()
            logit_pos_d.requires_grad = True
            logit_neg_d = logit_neg.detach()
            logit_neg_d.requires_grad = True
            diffs = F.sigmoid(logit_pos_d - logit_neg_d)
        else:
            logit_pos = pos_outputs.logits[:,1] - pos_outputs.logits[:,0]
            logit_neg = neg_outputs.logits[:,1] - neg_outputs.logits[:,0]

            logit_pos_d = logit_pos.detach()
            logit_pos_d.requires_grad = True
            logit_neg_d = logit_neg.detach()
            logit_neg_d.requires_grad = True
            diffs = F.sigmoid(logit_pos_d - logit_neg_d)
        
        all_diffs.append(diffs.cpu().detach())
        # Make the diff between the relevance of positive and negative along the path
        sum_outputs = torch.sum(diffs, dim=0)

        cross_scorer.zero_grad()
        sum_outputs.backward()
        (logit_pos * logit_pos_d.grad).sum().backward()
        for key, value in extractor.pos_outputs.items():
            if i == 0:
                # We don't care about this step as the first one is the baseline
                pass
            else:
                # Then, we compute the integral, which means we need to access the previous step's outputs 
                # and the current ones, from the extractor
                diff = (value.detach().cpu() - extractor.previous_pos_outputs[key]) # Diff between the current and previous outputs
                prod = diff * value.grad.data.detach().cpu() # Multiply this diff by the current gradient
                # Store the product and accumulate them along the path
                product_storage[key] = prod if i == 1 else product_storage[key] + prod 
        
    extractor.clear_items()
    extractor.remove_hooks()
 
    all_outputs = np.concatenate(all_diffs, axis=0)
    differences = [abs(all_outputs[i + 1] - all_outputs[i]) for i in range(len(all_outputs) - 1)]

    if compute_error:
        positive_errors = dict()
        for key in product_storage.keys():
           positive_errors[key] = _get_ig_error(product_storage[key], all_diffs[0][0], all_diffs[-1][-1], debug=False)
            
    gc.collect()
    torch.cuda.empty_cache()            

    if compute_error:
        return product_storage, positive_errors
    else:
        return product_storage, None

class AttributeInfoNCE(Task):
    __xpmid__="src.attribution_methods.attribution_infonce.attributeinfonce"

    cross_scorer: Param[Scorer]

    sampler: Param[Sampler]

    steps: Param[int]

    batch_size: Param[int]

    random : Param[Random]

    max_samples: Param[int]

    max_input_length: Param[int]

    baseline_method: Param[Baseline]

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    checkpoint_interval: Meta[int] = 1

    adaptive_sampling: Param[bool] = False

    compute_error: Meta[bool] = False

    ig_path: Annotated[Path, pathgenerator("ig")]

    iterator_path: Annotated[Path, pathgenerator("iterator")]

    def initialize(self, device):
        self.cross_scorer.initialize(ModuleInitMode.DEFAULT.to_options(self.random.state))
        self.cross_scorer.eval()
        self.cross_scorer.to(device)
        
        if self.cross_scorer.model.num_labels == 1:
            self.num_labels = 1
            self.activation_fct = lambda x, dim: F.sigmoid(x)
            self.get_pred_label = lambda x: int(x > 0.5)
        else:
            self.num_labels = 2
            self.activation_fct = lambda x, dim: F.softmax(x, dim=dim)
            self.get_pred_label = lambda x: int(torch.argmax(x, dim=1))

        self.tokenizer = self.cross_scorer.tokenizer

        self.embeddings = self.cross_scorer.model.get_input_embeddings()
        self.storage = list()
        self.error_storage = dict()
        self.sampler.initialize(self.random.state)
        self.sampler_iter = self.sampler.sampler_iter()

        self.skipping_iterator = SkippingIterator(self.sampler_iter)

    def reload_iterator(self)->int:
        last_count = 0
        iterator_files = sorted(glob.glob(f"{self.iterator_path}_step-*.json"))
        if iterator_files:
            last_iterator_file = iterator_files[-1]
            last_count = int(os.path.basename(last_iterator_file).split('_step-')[1].split('.json')[0])
            logging.info(f"Last iterator found for step: {last_count:06d}")
            with open(f"{self.iterator_path}_step-{last_count:06d}.json", 'r') as iterator_file:
                iterator_state = json.load(iterator_file)

            last_count = iterator_state["samples"]

            logging.info(f"Checking that storage exists for step: {last_count:06d}")
            if Path(f"{self.ig_path}_step-{last_count:06d}.pkl.gz").exists():
                # Iterator exists, corresponding storage also exists, so we restart from there
                logging.info(f"Restarting from step: {last_count:06d}")
                self.skipping_iterator.restore_state(iterator_state)
            else:
                # Iterator exists but not the storage, so we need to find the last storage saved
                storage_files = sorted(glob.glob(f"{self.ig_path}_step-*.pkl.gz"))
                if storage_files:
                    # Some storage files exist, we take the last one
                    last_storage_file = storage_files[-1]
                    last_count = int(os.path.basename(last_storage_file).split('_step-')[1].split('.pkl')[0])
                    logging.info(f"Last storage found for step: {last_count:06d}")
                    # and load the corresponding iterator state
                    with open(f"{self.iterator_path}_step-{last_count:06d}.json", 'r') as iterator_file:
                        iterator_state = json.load(iterator_file)
                    last_count = iterator_state["samples"]
                    logging.info(f"Restarting from step: {last_count:06d}")
                    self.skipping_iterator.restore_state(iterator_state)
                else:
                    logging.info("No storage file found to reload. Starting from scratch.")
                    last_count = 0

        else:
            logging.info("No iterator file found to reload. Starting from scratch.")

        if last_count != 0 and self.compute_error:
            # Reload error storage
            logging.info("Loading past error storage")
            with open(Path(f"{self.ig_path}_error_step-{last_count:06d}.pkl"), 'rb') as f:
                self.error_storage = pickle.load(f)

        return last_count
        
    def save_state(self, count: int):
        # Now we can save our new storages
        logging.info(f"Saving storage at step: {count:06d}")
        with open(Path(f"{self.ig_path}_step-{count:06d}.pkl.gz"), 'wb') as f:
            with gzip.GzipFile(fileobj=f, mode='wb') as gz:
                pickle.dump(self.storage, gz)

        if self.compute_error:
            # For the error file, we erase the old one and save the current one
            logging.info(f"Saving error storages at step: {count:06d}")
            error_pkl_files = glob.glob(f"{self.ig_path}_error_step-*.pkl")
            with open(Path(f"{self.ig_path}_error_step-{count:06d}.pkl"), 'wb') as f:
                pickle.dump(self.error_storage, f)
        
        iterator_state = self.skipping_iterator.state_dict()
        with open(f"{self.iterator_path}_step-{count:06d}.json", 'w') as iterator_file:
            # Saves both the skipping iterator to recover the dataset but also the count to keep track of the number of samples actually seen
            iterator_state.update({"samples": count}) 
            json.dump(iterator_state, iterator_file)

        # After having saved everything, we clear the storage to free up space
        self.storage.clear()
        
        # Cleaning up the previous error storage files from the directory
        for file in error_pkl_files:
            if Path(file).exists():
                os.remove(file)

    def execute(self):
        self.device.execute(self.device_execute)
        
    def device_execute(self, device_information: DeviceInformation):
        self.initialize(device_information.device)
        count = self.reload_iterator()

        with tqdm(
            total=self.max_samples, desc=f"Computing attributions over ({self.max_samples} samples)"
        ) as tqdm_epochs:
            tqdm_epochs.update(count)
            for record in iter(self.skipping_iterator):
                sample = PairwiseRecordWithTarget(
                    record.query, record.positive, record.negative, 1
                )
                pos_input = self.tokenizer(
                            sample.query[TextItem].text, 
                            sample.positive[TextItem].text,
                            max_length=self.max_input_length,
                            truncation=True,
                            padding="max_length", # To get every sample to the same size
                            return_attention_mask=True,
                            return_tensors="pt"
                        ).to(device_information.device)
                pos_embeds = self.embeddings(pos_input["input_ids"])
                neg_input = self.tokenizer(
                            sample.query[TextItem].text, 
                            sample.negative[TextItem].text,
                            max_length=self.max_input_length, # True max length is 512 but it gets too long to compute
                            truncation=True,
                            padding="max_length", # To get every sample to the same size
                            return_attention_mask=True,
                            return_tensors="pt"
                        ).to(device_information.device)
                neg_embeds = self.embeddings(neg_input["input_ids"])

                pos_label = self.get_pred_label(self.activation_fct(self.cross_scorer.model(inputs_embeds=pos_embeds, token_type_ids=pos_input["token_type_ids"], attention_mask=pos_input["attention_mask"]).logits, dim=-1))
                neg_label = self.get_pred_label(self.activation_fct(self.cross_scorer.model(inputs_embeds=neg_embeds, token_type_ids=neg_input["token_type_ids"], attention_mask=neg_input["attention_mask"]).logits, dim=-1))
               
                if pos_label == 1 and neg_label == 0:                        
                    pos_baseline_embeds = self.baseline_method.generate_baseline(self.tokenizer, pos_input["input_ids"], self.embeddings, device_information.device)
                    neg_baseline_embeds = self.baseline_method.generate_baseline(self.tokenizer, neg_input["input_ids"], self.embeddings, device_information.device)

                    pos_spans = get_token_types_spans(pos_input["input_ids"], self.tokenizer)

                    pos_conductance_per_layer, pos_error_per_layer = nig_infonce_full_model(
                        cross_scorer=self.cross_scorer,
                        pos_input_embeddings=pos_embeds,
                        pos_token_type_ids=pos_input["token_type_ids"],
                        pos_attention_mask=pos_input["attention_mask"],
                        pos_baseline_embeddings=pos_baseline_embeds,
                        neg_input_embeddings=neg_embeds,
                        neg_token_type_ids=neg_input["token_type_ids"],
                        neg_attention_mask=neg_input["attention_mask"],
                        neg_baseline_embeddings=neg_baseline_embeds,
                        num_labels=self.num_labels,
                        batch_size=self.batch_size,
                        steps=self.steps,
                        compute_error=self.compute_error,
                        adaptive_sampling=self.adaptive_sampling,
                    )

                    current_storage = dict()

                    for key in pos_conductance_per_layer.keys():
                        current_storage[key] = pos_conductance_per_layer[key].detach().cpu().numpy()

                        if count == 0:
                            if pos_error_per_layer is not None:
                                self.error_storage[key] = [float(pos_error_per_layer[key].detach().cpu().numpy())]
                        else:
                            if pos_error_per_layer is not None:
                                self.error_storage[key].append(float(pos_error_per_layer[key].detach().cpu().numpy()))

                    self.storage.append((pos_spans, current_storage))
                    
                    count += 1
                    tqdm_epochs.update(1)
                    if count % self.checkpoint_interval == 0 and count > 0:
                        logging.info(f"Saving storages and iterator at step: {count} (real position in the dataset: {self.skipping_iterator.position})")
                        self.save_state(count)

                if count >= self.max_samples:
                    logging.info(f"Stopping after {count} samples.")
                    break
        
        if count % self.checkpoint_interval != 0: # Means we got out of the loop but didn't save the last state of the storages
            logging.info(f"Saving storages and iterator at step: {count} (real position in the dataset: {self.skipping_iterator.position})")
            self.save_state(count)

        # logging.info(f"Average quality of the path between the baseline and the original input: {somme_diff / count}")
        if self.compute_error:
            with open(Path(f"{self.ig_path}_positive_error.pkl"), 'wb') as f:
                pickle.dump(self.error_storage, f)
    