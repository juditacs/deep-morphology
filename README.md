# deep-morphology

PyTorch deep learning models for experiments in morphology.

# Requirements

* Python 3
* PyTorch
* PyYaml

# Training

    python deep_morphology/train.py -c <path_to_YAML_config> --train <path_to_train_file> --dev <path_to_dev_file>

## Training file

Plain text and .gz are supported. Pipes are accepted as well, this is
especially useful for toy experiments:

    python deep_morphology/train.py -c <path_to_YAML_config> --train <( head -100 train_file) --dev <(head -100 dev_file)

Training and development files are expected to contain one sample-per-line with
the input and output separated by TAB. Symbols are separated by spaces. For
example a one sentence English-French parallel corpus would look like this
(tokenization may differ):

~~~
I am hungry <TAB> J' ai faim
~~~

Most of my experiments are character-level.
For example the training corpus for Hungarian instrumental case looks like this:

~~~
a l m a <TAB> a l m á v a l
k ö r t e <TAB> k ö r t é v e l
v i r á g <TAB> v i r á g g a l
~~~

In this case a single character is a symbol.

Test files are very similar except only the first column is used (if there are
more, the rest are ignored).

An experiment is described in a YAML configuration file. A toy example is available at `config/toy.yaml`.

Variable such as `${VAR}` are expanded using environment variables.

**WARNING** this is done manually since YAML does not support external variables.
My implementation may easily be exploited, do not run it as a web service or the like.

In the toy example only one such variable is used, you can set it with:

    mkdir -p experiments/toy
    export EXP_DIR=experiments

then you can run the experiment:

    python deep_morphology/train.py -c config/toy.yaml --train data/toy --dev data/toy

You should see a bunch of log messages:

* the train and validation loss printed after each epoch,
* the model saved after an epoch if the validation loss decreased to a file
  called `model.epoch_NNNN`, where `NNNN` is the epoch number,
* the current output to the toy evaluation set (listed in the variable `toy_eval` in
  the configuration file). I use this to make sure that the model is not
  complete garbage (the toy model will be garbage).

The experiment can be stopped any time with Ctrl+C. The best model will have
already been saved and other files such as result statistics are also saved upon exiting.

## Experiment directory

By default an empty subdirectory is created under `experiment_dir` with a
4-digit name and everything related to the experiment is saved into this
directory.

An experiment directory contains:

1. `config.yaml`: the final full configuration is saved here.
2. `result.yaml`: contains the train and val loss in each epoch and the experiment's timestamp and running time.
3. `vocab_src` and `vocab_tgt`: the source and target language vocabularies.
4. `model.epoch_N` and similar: model parameters after epoch N if the
   validation loss decreased compared to the current minimum.

