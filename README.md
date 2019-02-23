# Learning-to-learn-by-association
Tel Aviv University Deep Learning course project. Amit Henig &amp; Paz Ilan
This repository contains code for the project Learning to learn by Association for the Deep Learning course at TAU.

The code is derivded in part from https://github.com/haeusser/learning_by_association with our additional contributions added as described
in our project report (included in this repository).The machine learning framework used is tensorflow.

The core functions are implemented in backend.py inside the semisup class, but the training and evaluation is done in train_and_eval.py.

Before you get started:

1. Add mnist of fashion mnist data to the data folder and update the data_dirs.py files to thier location.

2. Add the path to learning to learn by association to your python path:

	For linux users (this is from the original repository, we did not work in linux so we don't know if it works):
	add the following to ~/.bashrc:
	export PYTHONPATH=/path/to/learning_to_learn_by_association:$PYTHONPATH

	For pycharm users:
	Go to the Main PyCharm setting, which will open up a separate window. In the left pane, choose Project:... > Project Interpreter.
	Now, in the main pane on the right, click the gear symbol next to the field for "Project Interpreter". Choose Show All in the menu 
	that pops up. Now in the final step, pick the interpreter you are using for this project and click on the tree symbol at the bottom 
	of the symbol list to the right of the window (hovering over the symbol reveals it as "Show paths for the selected interpreter"). 
	Add /path/to/learning_to_learn_by_association by clicking the "plus" symbol.

3. Choose a predefined configuration from the param_collection folder or create your own param folder and place it in the param folder and not under any sub folders!

After the above steps, in order to run our project simply call the tests_runner.py main.

Json param (in "") options and default values:
{
    "sup_per_class" : 10 ,# How many labeled samples per class to take, -1 means all
    "sup_seed" : 5 ,# Labeled example selection seed, -1 means random
    "sup_per_batch" : 10 ,# Batch size for the labeled set for each class
    "unsup_batch_size" : 100 ,# Batch size of the unlabeled samples
    "unsup_samples" : -1 ,# How many unlabeled samples to take, -1 means all
    "eval_interval" : 50 ,# Model evaluation interval, how many epochs between every progress save and print
    "learning_rate" : 1e-3 ,# Starting learning rate
    "decay_factor" : 0.33 ,# Factor for learning rate decay
    "decay_steps" : 5000 ,# Steps between each decay of learning rate
    "visit_weight" : 1.0 ,# Weight for the visit loss
    "max_steps" : 20000 ,# Total epochs in the training
    "logdir" : "./log/semisup_mnist" ,# Path to save the logs to
    "reg_M_weight" : 1e-3 ,# Weight for l2 regularization on Mnet
    # M_depth_struct is a struct holding the structure of Mnet, list of list of layers:
    # [neuron amount,activation,fc/conv (nothing=conv)]
    "M_depth_struct" : [[64, "sigmoid"],[32, "sigmoid"],[32, "sigmoid"],[16, "sigmoid"],[16, "sigmoid"]],
    # training_regime is a list of list of lists formatted:
    # [[[[vars to train],[losses to use]], ...],[num_epochs]]...]
    "training_regime" : [[[[["emb", "logit"], ["logit", "semisup"]]], 0]],
    "run_name" : "" ,# Name for the run to be saved under (for logs)
    "dataset" : "mnist" ,# Dataset name, needs data under data folder (we checked mnist is working and enabled fashion mnist)
    "architecture" : "mnist_model" ,# Model name to use, needs to be in the file architecture.py
    "collect_runtime_data" : False ,# Flag whether to collect runtime statistics (graph runtimes for tensorboard)
    "M_activation" : "None" ,# Which activation to choose for last Mnet layer
    "M_dropout" : 0 ,# Dropout prectange for Mnet
    "minimum_learning_rate" : 0 ,# Lowest allowable learning rate
    "triplet_margin" : 1.5 ,# Margin used for all triplet losses
    "reg_embedding" : 1e-3 ,# L2 Regulariztion weight value for the embedding network
    "logit_weight" : 1.0 ,# Weight for logit loss
    "Msup_weight" : 1.0 ,# Weight for our supervised comparison loss
    "triplet_hard_mining" : False ,# Whether to use hard mining for triplets or not
    "M_batch_norm" : True ,# Whether to use batch normalization in Mnet or not
    "save_model" : False # Save model checkpoint or not
}


Note that any parameter not mentioned in the JSON file will get its default value.


