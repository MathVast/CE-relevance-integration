<div align="center">

<h1>Relevance Integration study in Cross-Encoder Models</h1>
<div>
    <a href=https://scholar.google.com/citations?user=QGCo1PAAAAAJ&hl target='_blank'>Mathias Vast</a><sup>12</sup>&emsp;
    <a target='_blank'>Basile van Cooten</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.fr/citations?user=3gUQp6oAAAAJ&hl' target='_blank'>Laure Soulier</a><sup>1</sup>&emsp;
    <a href='https://www.piwowarski.fr' target='_blank'>Benjamin Piwowarski</a><sup>1</sup>&emsp;
</div>
<br>
<div>
    <sup>1</sup>Sorbonne Université, CNRS, ISIR, F-75005 Paris, France&emsp;<br>
    <sup>2</sup>ChapsVision, Paris, France&emsp;<br>
</div>
<br>

</div>

### Abstract 

> _Cross-encoders are the most effective but least efficient models in Information Retrieval, yet few studies have explored how they function or revealed the processes behind their relevance predictions. This gap can lead to suboptimal retrieval, especially out-of-domain, and efficiency issues with over-parameterized models. Our paper addresses two main limitations. First, the understudied overall relevance prediction process. Starting from the identification of the most important components in the model, we identify the flow of the information at the different stages of the process and between different input segments. We study the nature of the signals that compose this flow and that allow cross-encoders to predict the relevance of a passage to a query. Our work particularly focuses on the matching performed by the cross-encoders between queries and documents. Second, no work has looked into how differently cross-encoders behave with out-of-domain documents and/or queries. In this work, we conduct a series of interpretability experiments across both OOD and ID datasets on two cross-encoder models of different sizes.
Overall, the objective of this study is twofold: to strengthen what we currently know about the modeling of relevance by cross-encoders and to extend it by exploring the role of matching signals inside it._

### Description 

This code base is related to the paper "How Cross-Encoders Model Relevance in Information Retrieval?" by Mathias Vast, Basile Van Cooten, Laure Soulier & Benjamin Piwowarski.

At the moment, it contains all the code necessary to reproduce the experiments of the paper, i.e.:
- The Neuron Integrated Gradients (Sections 4 and 5)
- The ablation study (Section 5)
- The Information Bottleneck approach (Section 5)
- The attention patterns analysis (Section 6)
- The Linear Discriminant Analysis (Section 7)

### Installation
To install this repository, first ensure you have `git` and `uv` installed.

1.  Clone the repository and its submodules:
    ```bash
    git clone git@github.com:MathVast/CE-relevance-integration.git
    cd CE-relevance-integration
    ```

2.  Synchronize the Python dependencies using `uv`:
    ```bash
    uv sync
    ```

### Running experiments
[Experimaestro](https://github.com/experimaestro/experimaestro-python) is used to launch and monitor experiments.
Each technique has its own experimental plan and configuration file that needs to be combined to run properly.

A command example looks like:
`uv run experimaestro run-experiment --workdir /path/to/xps_folder_storage/ --file /path/to/technique_main_script.py /path/to/technique_config.yaml --run-mode DRY_RUN` (for actually launching the scripts, remove the `--run-mode` argument).

All the configuration files present in the repository are for MiniLM-v2 ([cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)). To switch to MonoBERT-large (though some experiments may require adapted requirements), just set the `ranker_id` to `castorini/monobert-large-msmarco` ([castorini/monobert-large-msmarco](https://huggingface.co/castorini/monobert-large-msmarco)).


### Acknowledgements
We depend on several key packages:
- [`experimaestro-python`](https://github.com/experimaestro/experimaestro-python) for experiment management.
- [`ir-datasets`](https://ir-datasets.com/) to access IR collections.

### Citation

The paper is currently under-review and the citation will be updated following the notifications.