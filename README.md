# Stack Attention: Improving the Ability of Transformers to Model Hierarchical Patterns

This repository contains the code for the paper
["Stack Attention: Improving the Ability of Transformers to Model Hierarchical Patterns"](https://arxiv.org/abs/2310.01749)
(DuSell and Chiang, 2024). It includes all of the code necessary to reproduce the
experiments and figures used in the paper, as well as a Docker image definition
that can be used to replicate the software environment it was developed in.

This code is partly based on the
[code](https://github.com/bdusell/nondeterministic-stack-rnn)
used for our earlier paper
["The Surprising Computational Power of Nondeterministic Stack RNNs"](https://arxiv.org/abs/2210.01343)
(DuSell and Chiang, 2023).

This repository includes PyTorch implementations of the following models:

* Transformer with
  [Nondeterministic Stack Attention](src/stack_attention/nondeterministic.py),
  a.k.a. "Tf+Nd" (introduced in this paper).
* Transformer with
  [Superposition Stack Attention](src/stack_attention/superposition.py),
  a.k.a. "Tf+Sup" (introduced in this paper).
* [Baseline Transformer](src/transformer_model/),
  a.k.a. "Tf"
  ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).
* [LSTM with differentiable vector PDA](src/stack_rnn_models/vector_nondeterministic_stack.py)
  (a.k.a. "LSTM+Nd" or "VRNS-RNN")
  ([DuSell and Chiang, 2023](https://arxiv.org/abs/2210.01343)).
* [LSTM with superposition stack](src/stack_rnn_models/joulin_mikolov.py)
  (a.k.a. "LSTM+Sup")
  ([Joulin and Mikolov, 2015](https://arxiv.org/abs/1503.01007)).
* [Baseline LSTM](src/torch_unidirectional/rnn.py)
  ([Hochreiter and Schmidhuber, 1997](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)).

All of the transformer models are available as both language models and
encoder-decoder models. Different types of attention layer can be mixed and
matched in any order.

## Directory Structure

* `data/`: Contains datasets used for experiments.
* `experiments/`: Contains scripts for reproducing all of the experiments and
  figures presented in the paper. Details below.
* `scripts/`: Contains helper scripts for setting up the software environment,
  building container images, running containers, installing Python packages,
  preprocessing data, etc. Instructions for using these scripts are below.
* `src/`: Contains source code for all models, training routines, plotting
  scripts, etc.
* `tests/`: Contains unit tests for the code under `src/`.

## Installation and Setup

In order to foster reproducibility, the code for this paper was developed and
run inside of a [Docker](https://www.docker.com/) container defined in the file
[`Dockerfile-dev`](Dockerfile-dev). To run this code, you can build the
Docker image yourself and run it using Docker. Or, if you don't feel like
installing Docker, you can simply use `Dockerfile-dev` as a reference for
setting up the software environment on your own system. You can also build
an equivalent [Singularity](https://sylabs.io/docs/#singularity) image which
can be used on an HPC cluster, where it is likely that Docker is not available
but Singularity is.

In any case, it is highly recommended to run most experiments on a machine with
access to an NVIDIA GPU so that they finish within a reasonable amount of time.
The exception to this is the experiments for the LSTM, LSTM+Sup, Tf, and Tf+Sup
models on the CFL language modeling tasks, which finish quite quickly in CPU
mode.

Note that when we ran our experiments, we used the base Docker image
`nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04`. However, support for the CUDA 10
image recently ended, and it is no longer available for download. Consequently,
we have updated the base image to
`nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04`. This should be compatible with
our code, but it has not been thoroughly tested.

### Using Docker

In order to use the Docker image, you must first
[install Docker](https://www.docker.com/get-started).
If you intend to run any experiments on a GPU, you must also ensure that your
NVIDIA driver is set up properly and install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

In order to automatically build the Docker image, start the container, and open
up a bash shell inside of it, run

    $ bash scripts/docker-shell.bash --build

After you have built the image once, there is no need to do so again, so
afterwards you can simply run

    $ bash scripts/docker-shell.bash

By default, this script starts the container in GPU mode, which will fail if
you are not running on a machine with a GPU. If you only want to run things in
CPU mode, you can run

    $ bash scripts/docker-shell.bash --cpu

### Using Singularity

If you use a shared HPC cluster at your institution, it might not support
Docker, but there's a chance it does support Singularity, which is an
alternative container runtime that is more suitable for shared computing
environments.

In order to run the code in a Singularity container, you must first obtain the
Docker image and then convert it to a `.sif` (Singularity image) file on a
machine where you have root access (e.g. your personal computer or
workstation). This requires installing both Docker and
[Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html)
on that machine. Assuming you have already built the Docker image according to
the instructions above, you can use the following to create the `.sif` file:

    $ bash scripts/build-singularity-image.bash

This will create the file `stack-attention.sif`. It is normal for this to take
several minutes. Afterwards, you can upload the `.sif` file to your HPC cluster
and use it there.

You can open a shell in the Singularity container using

    $ bash scripts/singularity-shell.bash

This will work on machines that do and do not have an NVIDIA GPU, although it
will output a warning if there is no GPU.

You can find a more general tutorial on Singularity
[here](https://github.com/bdusell/singularity-tutorial).

### Additional Setup

Whatever method you use to run the code (whether in a Docker container,
Singularity container, or no container), there are some additional setup and
preprocessing steps you need to run.

If you want to run the PTB language modeling experiments, you must obtain a
copy of the raw CD-ROM files (we are not able to distribute these due to
licensing restrictions). Make sure that these files can be found at
`data/ptb/`, either by copying the files or creating a symlink. Specifically,
only one sub-directory of the PTB distribution is needed:
`data/ptb/dist/treebank_3/parsed/mrg/wsj/` needs to have the contents of
`dist/treebank_3/parsed/mrg/wsj/` from the PTB distribution.

The following script will take care of some setup tasks for you (if you are
using a container, you must run this *inside the container shell*):

    $ bash scripts/setup.bash

Specifically, this script:

* Installs the Python packages required by our code, which will be stored in
  the local directory rather than system-wide. We use the package manager
  [Poetry](https://python-poetry.org/) to manage Python packages.
* Preprocesses the Penn Treebank language modeling dataset, assuming
  `data/ptb/dist/treebank_3/parsed/mrg/wsj` contains (or is a symlink
  to) the contents of `dist/treebank_3/parsed/mrg/wsj` in the raw PTB CD-ROM
  files.

## Running Code

All files under `src/` should be run using `poetry` so they have access to the
Python packages provided by the Poetry package manager. This means you should
either prefix all of your commands with `poetry run` or run `poetry shell`
beforehand to enter a shell with Poetry's virtualenv enabled all the time. You
should run both Python and Bash scripts with Poetry, because the Bash scripts
might call out to Python scripts. All Bash scripts under `src/` should be run
with `src/` as the current working directory.

All scripts under `scripts/` and `experiments/` should be run with the
top-level directory as the current working directory.

## Running Experiments

The [`experiments/`](experiments) directory contains scripts for reproducing
all of the experiments and plots presented in the paper. Some of these scripts
are intended to be used to submit jobs to a computing cluster. They should be
run outside of the container. You will need to edit the file
[`experiments/submit-job.bash`](experiments/submit-job.bash)
to tailor it to your specific computing cluster. Other scripts are for plotting
or printing tables and should be run inside the container.

### Context-Free Languages

The relevant scripts are under `experiments/cfl`.

* `submit_train_jobs.bash`: Train all models.
* `submit_generate_test_data_jobs.bash`: Generate the test sets.
* `submit_test_jobs.bash`: Evaluate the best models on the test sets.
* `plot_train_and_test.bash`: Generate the plots for the CFL experiments.
* `print_parameter_counts.bash`: Print the table of parameter counts.

### Language Modeling

As mentioned above, in order to generate the tokenized files for the Penn
Treebank language modeling task, you need to copy or symlink the contents
of `dist/treebank_3/parsed/mrg/wsj` from the original PTB distribution to
`data/ptb/dist/treebank_3/parsed/mrg/wsj`, then run `scripts/setup.bash`.

The relevant scripts are under `experiments/language-modeling`.

* `submit_preprocess_data_jobs.bash`: This must be run before training any
  models. This further preprocesses the tokenized data so it can be used for
  training and evaluation.
* `submit_limited_size_jobs.bash`: Train and evaluate all models.
* `print_limited_size_table.bash`: Print the table of perplexity scores.
* `print_cost_table.bash`: Print the table of computational cost.

### Machine Translation

The relevant scripts are under `experiments/machine-translation`.

Before running the machine translation experiments, the datasets must be
downloaded and preprocessed using these scripts:

* `submit_download_data_jobs.bash`: Download all datasets. This must be run
  first.
* `submit_preprocess_data_jobs.bash`: Preprocess all datasets. This must be run
  before training any models.
* `submit_varying_size_jobs.bash`: Train and evaluate all models.
* `print_varying_size_table.bash`: Print the table of scores.

## Citation

```bibtex
@inproceedings{dusell-chiang-2024-stack,
    title = "Stack Attention: Improving the Ability of Transformers to Model Hierarchical Patterns",
    author = "DuSell, Brian and Chiang, David",
    booktitle = "The Twelfth International Conference on Learning Representations",
    year = "2024",
    month = may,
    address = "Vienna, Austria",
    url = "https://openreview.net/forum?id=XVhm3X8Fum"
}
```
