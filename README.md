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
2. `result.yaml`: contains the train and val loss in each epoch and the
   experiment's timestamp and running time.
3. `vocab_*` and `vocab_*`: vocabularies for the model. Most models have a
   source and a target vocabulary but some models have more than two.
4. `model*` and similar: saved model(s). By default only the best (lowest dev
   loss) model is saved. If `overwrite_model` is set to False, every model that
   is better than the previous best one is saved as `model.epoch_N`, where `N`
   is the epoch number (starting from 0).

## Inference

   python deep_morphology/inference.py -e <EXPERIMENT DIR> -t <TEST FILE> > output

The inference script's only mandatory argument is the experiment directory. It
loads the configuration and the model from the directory and runs the model on
the test data. Test data may be provided via a file (`-t` option), otherwise it
is read from the standard input. The output is written to the standard output.

## Hyperparameter tuning

`train_many.py` runs several experiments after each other with parameters
uniformly sampled from predefined ranges.

Usage:

    python deep_morphology/train_many.py --config <YAML config_file> --param-ranges <YAML ranges file> -N 10 --train <train_file> --dev <dev_file>

Explanation:

* `config`: base configuration file. Loaded before each experiment.
* `param-ranges`: file containing the predefined parameter ranges, see the
  example below.
* `N`: number of experiments to run.

### Parameter ranges file

This file is a list of key-value pairs, where the keys are the parameter name
(they must be the same as in the Config object) and the values are a list of
possible parameter values. For example:

~~~yaml
hidden_size_src: [128, 256, 512, 1024]
num_layers_src: [1, 2, 3]
dropout: [0, 0.2, 0.5, 0.8]
embedding_size_src: [10, 20, 30]
batch_size: [32, 64, 128]
~~~

This example can be found
[here](https://github.com/juditacs/deep-morphology/blob/master/config/tagging/param_ranges.yaml).

Each parameter is replaced in the base config with a randomly samples value
from the parameter ranges file.

# Adding a new dataset

* Create a new source file in `deep_morphology/data`. Let's name it
  `dummy_data.py`
* Create a data class that inherits `BaseDataset`.
* Create a `recordclass` (mutable version of `collections.namedtuple`) which
  lists the fields of this class.
* Define the labeled and unlabeled dataset classes (see below).
* Add these to `deep_morphology/data/__init__.py`

## Dataset class

The dataset class must define 3 class attributes:

1. `unlabeled_data_class`: this is the name of the corresponding unlabeled dataset class. The name should be a string since the class does not exits yet.
2. `data_recordclass`: he recordclass corresponding to this dataset class. This should be the type itself (not a string).
3. `constants`: list of constants that the vocabularies should define. Note that these used by `load_or_create_vocab` which can be redefined if necessary (for example different fields of the dataset should use different constants).

The class must define the following functions:

1. `extract_sample_from_line`: extracts a single sample from a line and returns and instance of the class' recordclass.
2. `

The class may override the following functions:

1. `ignore_sample`: returns `True` if a sample should be skipped.

### Unlabeled dataset class

This class is used during inference.  This should inherit from the labeled
dataset class.  You most likely need to override `extract_sample_from_line` to
read unlabeled samples.


# Adding a new model

Models should inherit from `BaseModel` and define an `__init__`, and a `forward` function.
