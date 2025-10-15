# Relevance Integration study in Cross-Encoder Models

This code base is related to the paper "How Cross-Encoders Model Relevance in Information Retrieval?" by Mathias Vast, Basile Van Cooten, Laure Soulier & Benjamin Piwowarski.

At the moment, it contains all the code necessary to reproduce the experiments of the paper, i.e.:
- The Neuron Integrated Gradients (Sections 4 and 5)
- The ablation study (Section 5)
- The Information Bottleneck approach (Section 5)
- The attention patterns analysis (Section 6)
- The Linear Discriminant Analysis (Section 7)

## Getting started
You can use the `pyproject.toml` file after cloning the repo locally to set up the environment.

## Using experimaestro to launch experiments
This code base rely on [experimaestro](https://experimaestro-python.readthedocs.io/en/latest/) to manage the experimental plan. Each technique has its own experimental plan and configuration file that needs to be combined to run properly.

A command example looks like:
`experimaestro run-experiment --workdir /path/to/xps_folder_storage/ --file /path/to/technique_main_script.py /path/to/technique_config.yaml --run-mode DRY_RUN` (for actually launching the scripts, remove the `--run-mode` argument).

All the configuration files present in the repository are for MiniLM-v2 ([cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)). To switch to MonoBERT (though some experiments may require adapted requirements), just set the `ranker_id` to `castorini/monobert-large-msmarco` ([castorini/monobert-large-msmarco](https://huggingface.co/castorini/monobert-large-msmarco)).

## Contact

Feel free to contact either Laure, Benjamin or Mathias at (name).(surname)@isir.upmc.fr