from collections import defaultdict
import itertools
from pathlib import Path
import re
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import LogLocator, MaxNLocator
from matplotlib import cm
import pandas as pd

INPUT_PART_TO_POSITION = {"cls": 0, "query": 1, "sep_1": 2, "document": 3, "sep_2": 4}
INPUT_PART_TO_ANNOTATION = {"cls": "C", "query": "Q", "sep_1": "S1", "document": "D", "sep_2": "S2"}


def filter_module(module: str, keywords: List[str]):
    return any(word in module for word in keywords)

def is_pair_interesting(pair, regex: Optional[List[str]] = None):
    interesting_layers = ["self.query", "self.value", "self.key", "attention_probs", "intermediate.dense", "output.dense"] # As we start from the embeddings, we don't want to include them in our analysis
    name, _ = pair
    if any(word in name for word in interesting_layers):
        if regex is not None:
            return filter_module(name, regex)
        else:
            return True
    else:
        return False


def generate_layerwise_visualization(pkl_file: Path, num_layers: int, num_heads: int, hidden_dim: int, percentage: float, type: str, model: str) -> None:
    """
    Generate a graphic illustrating the repartition of the most ablated neurons amongst the modules of each layer of the model.

    :param Path pkl_file: The pruning scheme for which you want to obtain the visualization.
    """
    cmap = cm.get_cmap("YlOrRd")
    intermediate_color = cmap(0.3)  # lighter yellow-orange
    attention_color = cmap(0.8) 

    with open(pkl_file, 'rb') as f:
        sorted_layers_by_ablated_neurons = pickle.load(f)

    attention_probs_pattern = re.compile(r'(bert\.encoder\.layer\.\d+\.attention\.self\.attention_probs)(?:\.\d+)?')
    data = defaultdict(int)
    for layer_idx in range(num_layers):
        # Add intermediate dense
        intermediate_key = f"bert.encoder.layer.{layer_idx}.intermediate.dense"
        data[intermediate_key] = 0

        # Add all attention head keys (normalized version)
        normalized_attention_key = f"bert.encoder.layer.{layer_idx}.attention.self.attention_probs"
        data[normalized_attention_key] = 0  # aggregate under normalized key

    # Fill actual values from the model
    for key, values in sorted_layers_by_ablated_neurons.items():
        
        match = attention_probs_pattern.match(key)
        if match:
            count = len(values['all'][0]) if isinstance(values['all'][0], np.ndarray) else len(values['all'])
            normalized_key = match.group(1)
            data[normalized_key] += count
        else:
            count = len(values['all'])
            data[key] = count

    # Convert back to a regular dict if needed
    data = dict(data)

    layerwise_dict = dict()
    for name, neuron_nb in data.items():
        splitted_name = name.split(".")
        if splitted_name[3] not in layerwise_dict.keys():
            # First time we meet this layer, init the entry
            layerwise_dict[splitted_name[3]] = {"intermediate": 0, "attention_probs": 0}

        if splitted_name[4] == "attention":
            layerwise_dict[splitted_name[3]]["attention_probs"] += neuron_nb
        else:
            layerwise_dict[splitted_name[3]]["intermediate"] += neuron_nb
        
    dico_tuples = dict()
    sorted_layerwise_dict = sorted(layerwise_dict.items(), key=lambda x:int(x[0]), reverse=False)
    for (_, dico) in sorted_layerwise_dict:
        for key, value in dico.items():
            if key not in dico_tuples.keys():
                dico_tuples[key] = tuple([value])
            else:
                dico_tuples[key] += tuple([value])

    # Graphic generation
    
    plt.clf()

    layers = [f"{idx}" for idx in range(len(layerwise_dict))]
    x = np.arange(len(layers))
    width = 0.4  # wider bars for better visibility

    fig, ax1 = plt.subplots(layout='constrained')
    ax2 = ax1.twinx()  # second y-axis

    # Separate the values
    intermediate_values = dico_tuples.get("intermediate", ())
    attention_values = dico_tuples.get("attention_probs", ())

    # Plot bars
    rects1 = ax1.bar(x - width/2, intermediate_values, width, label='Intermediate', color=intermediate_color)
    rects2 = ax2.bar(x + width/2, attention_values, width, label='Attention Heads', color=attention_color)

    # ---- Log scale setup ----
    ax1.set_yscale('log')

    # Set common axis limits and ticks across plots
    log_bottom = 1  # or the smallest expected non-zero value
    log_top = hidden_dim  # or a fixed upper bound you choose

    ax1.set_ylim(bottom=log_bottom, top=log_top)
    ax1.set_yticks([1, 10, 100, 1000, hidden_dim])  # customize as needed
    ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))

    # Set linear scale for attention axis with fixed range
    ax2.set_ylim(0, num_heads)
    ax2.set_yticks(np.linspace(0, num_heads, 5))  # evenly spaced ticks (adjust as needed)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # X-axis setup
    ax1.set_xticks(x)
    ax1.set_xlim(-width, len(layers) - 1 + width)
    ax1.set_xticklabels(layers, rotation=0)
    ax1.set_xlabel('Layer Index')

    # Y-axis labels
    ax1.set_ylabel('Nb Intermediate Neurons per Layer', color="black")
    ax2.set_ylabel('Nb Attention Heads per Layer', color="black")

    # Legends
    fig.legend([rects1, rects2], ['Intermediate', 'Attention Probs'], loc='upper left', bbox_to_anchor=(0.1, 0.9), ncols=2)

    # Figure size
    fig.set_figheight(5)
    fig.set_figwidth(13)

    plt.title(f"Distribution of the top {percentage*100}% neurons in the model (ordered by their NIG score)")
    # plt.savefig(f"{model}_layerwise_viz_{type}_{percentage}.pdf", format='pdf')
    plt.show()


def generate_layerwise_visualization_v2(pkl_file: Path, num_layers: int, num_heads: int, hidden_dim: int, percentage: float, type: str, model: str) -> None:
    """
    Generate a graphic illustrating the repartition of the most ablated neurons amongst the modules of each layer of the model.

    :param Path pkl_file: The pruning scheme for which you want to obtain the visualization.
    """
    cmap = cm.get_cmap("YlOrRd")
    intermediate_color = cmap(0.3)
    attention_color = cmap(0.8)

    with open(pkl_file, 'rb') as f:
        sorted_layers_by_ablated_neurons = pickle.load(f)

    attention_probs_pattern = re.compile(r'(bert\.encoder\.layer\.\d+\.attention\.self\.attention_probs)(?:\.\d+)?')
    data = defaultdict(int)

    for layer_idx in range(num_layers):
        intermediate_key = f"bert.encoder.layer.{layer_idx}.intermediate.dense"
        normalized_attention_key = f"bert.encoder.layer.{layer_idx}.attention.self.attention_probs"
        data[intermediate_key] = 0
        data[normalized_attention_key] = 0

    for key, values in sorted_layers_by_ablated_neurons.items():
        match = attention_probs_pattern.match(key)
        if match:
            count = len(values['all'][0]) if isinstance(values['all'][0], np.ndarray) else len(values['all'])
            normalized_key = match.group(1)
            data[normalized_key] += count
        else:
            count = len(values['all'])
            data[key] = count

    data = dict(data)
    layerwise_dict = dict()
    for name, neuron_nb in data.items():
        splitted_name = name.split(".")
        layer = splitted_name[3]
        if layer not in layerwise_dict:
            layerwise_dict[layer] = {"intermediate": 0, "attention_probs": 0}

        if splitted_name[4] == "attention":
            layerwise_dict[layer]["attention_probs"] += neuron_nb
        else:
            layerwise_dict[layer]["intermediate"] += neuron_nb

    dico_tuples = dict()
    sorted_layerwise_dict = sorted(layerwise_dict.items(), key=lambda x: int(x[0]))
    for (_, dico) in sorted_layerwise_dict:
        for key, value in dico.items():
            if key not in dico_tuples:
                dico_tuples[key] = tuple([value])
            else:
                dico_tuples[key] += tuple([value])

    # ---- Normalization ----
    intermediate_values = np.array(dico_tuples.get("intermediate", ()), dtype=np.float32)
    attention_values = np.array(dico_tuples.get("attention_probs", ()), dtype=np.float32)

    intermediate_percentages = (intermediate_values / hidden_dim)
    attention_percentages = (attention_values / num_heads)

    # ---- Plotting ----
    plt.clf()
    layers = [f"{idx}" for idx in range(len(layerwise_dict))]
    x = np.arange(len(layers))
    width = 0.4

    fig, ax = plt.subplots(layout='constrained')

    rects1 = ax.bar(x - width/2, intermediate_percentages, width, label='Intermediate', color=intermediate_color)
    rects2 = ax.bar(x + width/2, attention_percentages, width, label='Attention Heads', color=attention_color)

    # Y-axis formatting
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylabel('Percentage of Neurons Pruned per Layer', color='black')

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel('Layer Index')

    # Legends
    ax.legend(loc='upper left', ncols=2)

    # Figure size
    fig.set_figheight(5)
    fig.set_figwidth(13)

    plt.title(f"Distribution of the top {percentage*100}% neurons in the model (ordered by their NIG score)")
    plt.savefig(f"{model}_layerwise_viz_{type}_{percentage}.pdf", format='pdf')
    plt.show()


def generate_layerwise_visualization_v3(pkl_files, num_layers: int, num_heads: int, hidden_dim: int, percentage: float, types: Optional[List[str]] = None, model: Optional[str] = None) -> None:
    """
    Generate a single graphic from multiple pruning-scheme pickles. Each pickle's
    layerwise curves (intermediate vs attention) are plotted on the same axes.

    - `pkl_files` can be a single Path/str or a list of Paths/strs.
    - `types` (optional) is a list of type labels (one per pickle) used for
      styling the curves (e.g. 'msmarco_intersection'). If not provided, the
      function will try to infer a label from the filename.

    Colors for `intermediate` and `attention` are kept consistent across all
    pickles; the line style / marker is varied per `type` so curves remain
    distinguishable.
    """
    # normalize pkl_files and types
    if not isinstance(pkl_files, (list, tuple)):
        pkl_files = [pkl_files]
    if types is None:
        types = []
        for p in pkl_files:
            stem = Path(p).stem
            # try to extract trailing type after last underscore
            if "_" in stem:
                types.append(stem.split("_")[-1] if stem.split("_")[-1] else stem)
            else:
                types.append(stem)
    else:
        if not isinstance(types, (list, tuple)):
            types = [types]

    cmap = cm.get_cmap("YlOrRd")
    intermediate_color = cmap(0.3)
    attention_color = cmap(0.8)

    # collect series
    all_intermediate = []
    all_attention = []
    inferred_types = []

    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as f:
            sorted_layers_by_ablated_neurons = pickle.load(f)

        counts = get_component_counts(sorted_layers_by_ablated_neurons, num_layers)
        inter = [counts[str(i)]["intermediate"] for i in range(num_layers)]
        attn = [counts[str(i)]["attention_probs"] for i in range(num_layers)]

        all_intermediate.append(np.array(inter, dtype=np.float32))
        all_attention.append(np.array(attn, dtype=np.float32))
        inferred_types.append(types[idx] if idx < len(types) else f"series_{idx}")

    # ---- Plotting ----
    plt.clf()
    layers = [f"{i}" for i in range(num_layers)]
    x = np.arange(len(layers))

    fig, ax1 = plt.subplots(layout='constrained')
    ax2 = ax1.twinx()

    # draw each series with consistent colors but varying style
    handles = []
    labels = []
    import matplotlib.lines as mlines

    def _short_label(name: str) -> str:
        tlow = name.lower()
        # kind: intersection -> I, fusion/merge -> F
        if 'intersection' in tlow:
            kind = 'I'
        elif 'fusion' in tlow or 'merge' in tlow:
            kind = 'F'
        else:
            kind = name[0].upper()

        # dataset: msmarco -> M, ood -> O, all -> A
        if 'msmarco' in tlow:
            ds = 'M'
        elif 'ood' in tlow:
            ds = 'O'
        elif 'all' in tlow:
            ds = 'A'
        else:
            # fallback: take initials of words
            parts = [p for p in re.split(r"[^A-Za-z]+", name) if p]
            ds = ''.join([p[0].upper() for p in parts]) or 'U'

        return f"{kind}_{ds}"

    for series_inter, series_attn, t in zip(all_intermediate, all_attention, inferred_types):
        # choose linestyle/marker by dataset family and keep ALL_intersection
        # and ALL_fusion distinct instead of collapsing both to 'all'
        tlow = t.lower()
        if 'msmarco' in tlow:
            linestyle = '-'
            marker = 'o'
        elif 'ood' in tlow:
            linestyle = '--'
            marker = 's'
        elif 'all_intersection' in tlow:
            linestyle = '-'
            marker = '^'
        elif 'all_fusion' in tlow:
            linestyle = '--'
            marker = 'v'
        elif 'all' in tlow:
            # generic fallback for any other 'all' labels
            linestyle = '--'
            marker = '^'
        else:
            linestyle = '--'
            marker = 'D'

        # increase marker size a bit for better visibility
        l1, = ax1.plot(x, series_inter, linestyle=linestyle, marker=marker, linewidth=2.0, markersize=8, color=intermediate_color, alpha=0.95)
        l2, = ax2.plot(x, series_attn, linestyle=linestyle, marker=marker, linewidth=2.0, markersize=8, color=attention_color, alpha=0.95)

        # Add legend entries (one label per type, showing both components)
        abbr = _short_label(t)
        # show abbreviation as LaTeX math (e.g. $I_M$) with component text
        handles.append(mlines.Line2D([], [], color=intermediate_color, linestyle=linestyle, marker=marker))
        labels.append(f"${abbr}$ (Intermediate)")
        handles.append(mlines.Line2D([], [], color=attention_color, linestyle=linestyle, marker=marker))
        labels.append(f"${abbr}$ (Attention)")

    # Left axis formatting (intermediate)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1, top=max(hidden_dim, 10))
    ax1.set_ylabel('Nb Intermediate Neurons per Layer', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Right axis formatting (attention)
    ax2.set_ylim(0, max(num_heads, 1))
    ax2.set_ylabel('Nb Attention Heads per Layer', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel('Layer Index')
    ax1.grid(True, alpha=0.25, linestyle='--')

    # Legend inside the main axes so it doesn't overflow the figure
    ax1.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), ncol=2, fontsize=12, framealpha=0.9)

    # Figure size
    fig.set_figheight(5)
    fig.set_figwidth(13)

    plt.title(f"Distribution of the top {percentage*100}% neurons in the model (ordered by their NIG score)")
    # save a combined filename if model provided
    if model is None:
        model = "model"
    safe_types = "_".join([re.sub(r"[^0-9a-zA-Z]+", "", t) for t in inferred_types])
    plt.savefig(f"{model}_layerwise_viz_combined_{safe_types}_{percentage}.pdf", format='pdf')
    plt.show()


# def generate_attention_head_heatmap_from_gradients(pkl_file: Path, layer_nb: int) -> None:
#     """
#     Generate a heatmap illustrating the number of neurons in each attention head for each layer of the model.

#     :param Path pkl_file: The pruning scheme for which you want to obtain the visualization.
#     """
#     with open(pkl_file, 'rb') as f:
#         nig_model = pickle.load(f)

#     nig_model = dict(filter(functools.partial(is_pair_interesting, regex=["attention_probs"]), nig_model.items()))

#     attention_probs_keys = [key for key in nig_model.keys() if "attention_probs" in key]
#     # As the attention_probs module is a 4D tensors with an additional dimensio for the attention_heads
#     # we split it into attention_heads 3D tensors in order for it to match the shape of the others
#     for key in attention_probs_keys:
#         if "attention_probs" in key:
#             attention_probs = nig_model.pop(key)
#             if isinstance(attention_probs, dict):
#                 for direction, values in attention_probs.items():
#                     for n, attention_head in enumerate(values):
#                         if key + f'.{n}' in nig_model.keys():
#                             nig_model[key + f'.{n}'][direction] = torch.tensor(attention_head) # Little trick to tranform every tensor to the same shape
#                         else:
#                             nig_model[key + f'.{n}'] = {direction: torch.tensor(attention_head)}
#             elif isinstance(attention_probs, np.ndarray):
#                 for n in range(attention_probs.shape[0]):
#                     if key + f'.{n}' in nig_model.keys():
#                         nig_model[key + f'.{n}'] = attention_probs[n]
#                     else:
#                         nig_model[key + f'.{n}'] = attention_probs[n]
                        
#     importance_matrix = torch.zeros((12,5,5))
#     for attention_head in range(0, 12):
#         for couple in itertools.product(input_part_to_position.keys(), repeat=2):
#             importance_matrix[attention_head,input_part_to_position[couple[0]],input_part_to_position[couple[1]]]  = nig_model[f"bert.encoder.layer.{layer_nb}.attention.self.attention_probs.{attention_head}"][f"{couple[0]}_{couple[1]}"]
    
#     normalized_importance_matrix = torch.nn.functional.normalize(importance_matrix, dim = (1,2)) 
#     fig, axes = plt.subplots(4, 4, figsize=(20, 20))
#     axes = axes.flatten()
#     for idx, attention_head in enumerate(normalized_importance_matrix):
#         axes[idx].imshow(attention_head.numpy(), cmap="RdYlBu_r", vmin=-1, vmax=1)
#         axes[idx].set_xticks(np.arange(len(input_part_to_position.keys())), labels=input_part_to_position.keys())
#         axes[idx].set_yticks(np.arange(len(input_part_to_position.keys())), labels=input_part_to_position.keys())
#         axes[idx].set_xlabel(f'Attention head: {idx}')
#         for i in range(len(attention_head)):
#             for j in range(len(attention_head[i])):
#                 text = axes[idx].text(j, i, f"{attention_head[i, j].item():.2f}",
#                             ha="center", va="center", color="black")

#     plt.title('Importance Matrix Heatmap')
#     plt.show()

def get_component_counts(raw_dict, num_layers=12):
    attention_probs_pattern = re.compile(r'(bert\.encoder\.layer\.\d+\.attention\.self\.attention_probs)(?:\.\d+)?')
    data = defaultdict(int)
    for layer_idx in range(num_layers):
        # Add intermediate dense
        intermediate_key = f"bert.encoder.layer.{layer_idx}.intermediate.dense"
        data[intermediate_key] = 0

        # Add all attention head keys (normalized version)
        normalized_attention_key = f"bert.encoder.layer.{layer_idx}.attention.self.attention_probs"
        data[normalized_attention_key] = 0  # aggregate under normalized key

    # Fill actual values from the model
    for key, values in raw_dict.items():
        
        match = attention_probs_pattern.match(key)
        if match:
            count = len(values['all'][0]) if isinstance(values['all'][0], np.ndarray) else len(values['all'])
            normalized_key = match.group(1)
            data[normalized_key] += count
        else:
            count = len(values['all'])
            data[key] = count

    # Build structured dict: layer -> {component -> count}
    layerwise = defaultdict(lambda: {"intermediate": 0, "attention_probs": 0})

    for key, count in data.items():
        split = key.split(".")
        layer = split[3]
        component = "attention_probs" if split[4] == "attention" else "intermediate"
        layerwise[layer][component] += count

    return layerwise  # e.g., {"0": {"intermediate": 512, "attention_probs": 4}, ...}

def normalize_dict(raw_dict, num_layers=12):
    """Normalizes and groups the raw dictionary like in your parsing logic, 
    but retains sets of neuron indices instead of counts."""
    attention_probs_pattern = re.compile(r'(bert\.encoder\.layer\.\d+\.attention\.self\.attention_probs)(?:\.\d+)?')
    data = defaultdict(set)

    for layer_idx in range(num_layers):
        intermediate_key = f"bert.encoder.layer.{layer_idx}.intermediate.dense"
        normalized_attention_key = f"bert.encoder.layer.{layer_idx}.attention.self.attention_probs"
        data[intermediate_key] = set()
        data[normalized_attention_key] = set()

    for key, values in raw_dict.items():
        match = attention_probs_pattern.match(key)
        if match:
            normalized_key = match.group(1)
            if values['all'][0] == 0:
                indices = [int(key[-1])]
            data[normalized_key].update(indices)
        else:
            indices = values['all']
            data[key].update(indices)

    # Group by (layer, component)
    layerwise = defaultdict(lambda: {"intermediate": set(), "attention_probs": set()})

    for name, index_set in data.items():
        splitted = name.split(".")
        layer = splitted[3]
        if splitted[4] == "attention":
            layerwise[layer]["attention_probs"].update(index_set)
        else:
            layerwise[layer]["intermediate"].update(index_set)

    return layerwise  # Dict[layer][component] = set()

def compute_agreement_rate(dict1, dict2, num_layers=12):
    """
    Compute agreement rate between two normalized dictionaries of ablated indices.

    Agreement is computed as Jaccard similarity: |A ∩ B| / |A ∪ B| for each (layer, component) pair.

    If both are missing the key → agreement = None.
    If only one is missing → agreement = 0.0 (i.e., ∅ vs non-empty).
    """
    all_keys = set()

    for i in range(num_layers):
        for component in ["intermediate", "attention_probs"]:
            all_keys.add((str(i), component))

    agreement = dict()

    for key in all_keys:
        set1 = dict1.get(key[0], {}).get(key[1], None)
        set2 = dict2.get(key[0], {}).get(key[1], None)

        if not set1 and not set2:
            agreement[key] = None
        else:
            s1 = set1 if set1 is not None else set()
            s2 = set2 if set2 is not None else set()
            union = s1 | s2
            inter = s1 & s2
            agreement[key] = len(inter) / len(union) if union else None # identical empty sets

    return agreement

def plot_comparison_with_agreement(raw1, raw2, num_layers, num_heads, hidden_dim, percentage, model_name):
    counts1 = get_component_counts(raw1, num_layers)
    counts2 = get_component_counts(raw2, num_layers)
    agreement = compute_agreement_rate(normalize_dict(raw1), normalize_dict(raw2), num_layers)

    layers = [str(i) for i in range(num_layers)]
    x = np.arange(len(layers))
    width = 0.2

    # Colors
    cmap = cm.get_cmap("YlOrRd")
    inter1_color = cmap(0.3)
    attn1_color = cmap(0.8)
    inter2_color = cmap(0.1)
    attn2_color = cmap(0.6)

    # Extract values per layer
    inter1 = [counts1[str(i)]["intermediate"] for i in range(num_layers)]
    attn1 = [counts1[str(i)]["attention_probs"] for i in range(num_layers)]
    inter2 = [counts2[str(i)]["intermediate"] for i in range(num_layers)]
    attn2 = [counts2[str(i)]["attention_probs"] for i in range(num_layers)]

    # Prepare plot
    plt.clf()
    fig, ax1 = plt.subplots(layout="constrained")
    ax2 = ax1.twinx()

    # Plot bars
    r1 = ax1.bar(x - 1.5 * width, inter1, width, label='Intermediate 1', color=inter1_color)
    r2 = ax1.bar(x - 0.5 * width, inter2, width, label='Intermediate 2', color=inter2_color)
    r3 = ax2.bar(x + 0.5 * width, attn1, width, label='Attention 1', color=attn1_color)
    r4 = ax2.bar(x + 1.5 * width, attn2, width, label='Attention 2', color=attn2_color)

    # Label agreement rate per bar group
    for i, layer in enumerate(layers):
        for comp, x_offset_left, val1, val2 in [
            ("intermediate", -width, inter1[i], inter2[i]),
            ("attention_probs", +width, attn1[i], attn2[i]),
        ]:
            key = (layer, comp)
            # Show label only if at least one bar is visible (non-zero)
            if (val1 + val2) > 0:
                # If key is missing from agreement dict, default to None
                agree = agreement.get(key, None)
                if agree is not None:
                    ax = ax1 if comp == "intermediate" else ax2
                    max_height = max(val1, val2)
                    y_offset = max_height + (5 if comp == "intermediate" else 0.3)
                    ax.text(
                        x[i] + x_offset_left,
                        y_offset,
                        f"{agree:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                        rotation=0,
                        color="black"
                    )

    # Log scale and limits
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0, top=100)
    ax1.set_yticks([1, 10, 100])
    ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))

    ax2.set_ylim(0, num_heads)
    ax2.set_yticks(np.linspace(0, num_heads, 5))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xticks(x)
    ax1.set_xlim(-0.5, len(layers)-0.5)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer Index")

    ax1.set_ylabel("Intermediate Neurons")
    ax2.set_ylabel("Attention Heads")

    # fig.legend([r1, r2, r3, r4], [r"Intermediate: $I_O$", r"Intermediate: $I_A$", r"Attention Heads: $I_O$", r"Attention Heads: $I_A$"], loc='upper left', bbox_to_anchor=(0.05, 0.92), ncols=2)
    leg1 = ax1.legend(
        [r1, r2],
        [r"Intermediate: $I_O$", r"Intermediate: $I_A$"],
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        fontsize=9,
    )

    # Second legend: Attention Heads (right)
    leg2 = ax2.legend(
        [r3, r4],
        [r"Attention Heads: $I_O$", r"Attention Heads: $I_A$"],
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        fontsize=9,
    )

    # Add both legends to the figure
    fig.add_artist(leg1)
    fig.add_artist(leg2)

    fig.set_figheight(4)
    fig.set_figwidth(14)
    plt.title(fr"Comparison of ablation schemes $I_O$ and $I_A$ at $t={percentage*100:.0f}\%$")
    plt.savefig(f"comparisons_intersection_all_ood_{model_name}.pdf", format='pdf')
    plt.show()

class LayerHook():
    """Register layer input"""
    def __init__(self, ix: int):
        self.layer_ix = ix
        self.handle = None
        
    def __call__(self, module, args):
        hidden_states, *_ = args
        self.hidden_states = hidden_states

    def close(self):
        if self.handle:
            print("Removing handle for layer", self.layer_ix)
            self.handle.remove()
        

def process_attention(attention_data: pd.DataFrame, duplicate_token_data: pd.DataFrame, entropy_data: pd.DataFrame) -> pd.DataFrame:
    # For each layer, compute the mean per query and relevance
    data_qid = attention_data.groupby(["layer", "head", "rel", "qid"]).mean(numeric_only=True).reset_index()
    data_duplicate_qid = duplicate_token_data.groupby(["layer", "head", "rel", "qid"]).mean(numeric_only=True).reset_index()
    data_entropy_qid = entropy_data.groupby(["layer", "head", "rel", "qid"]).mean(numeric_only=True).reset_index()
    data_qid = data_qid.merge(data_duplicate_qid, on=["layer", "head", "rel", "qid"])
    data_qid = data_qid.merge(data_entropy_qid, on=["layer", "head", "rel", "qid"])

    # partition the data
    d_weak = data_qid[data_qid.rel == -1].drop(columns=["rel"])
    d_hard = data_qid[data_qid.rel == 0].drop(columns=["rel"])
    d_rel = data_qid[data_qid.rel == 1].drop(columns=["rel"])

    # and merge...
    data_by_query = d_weak.merge(d_hard, on=["layer", "head", "qid"], suffixes=("_weak", "_hard"))
    data_by_query = data_by_query.merge(d_rel, on=["layer", "head", "qid"])

    for pf, cn in itertools.product(["ent_", "dup_", ""], ["q2d", "d2q"]):
        data_by_query[f"{pf}{cn}_delta_weak"] = data_by_query[f"{pf}{cn}"] - data_by_query[f"{pf}{cn}_weak"]
        data_by_query[f"{pf}{cn}_delta_hard"] = data_by_query[f"{pf}{cn}"] - data_by_query[f"{pf}{cn}_hard"]
        data_by_query[f"{pf}{cn}_delta_hard_weak"] = data_by_query[f"{pf}{cn}_hard"] - data_by_query[f"{pf}{cn}_weak"]

    return data_by_query