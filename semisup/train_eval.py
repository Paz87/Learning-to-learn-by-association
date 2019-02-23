"""
Revised version of mnist_train_eval.py from https://github.com/haeusser/learning_by_association for learning to learn by association
with the following Copyright header:
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


- Association-based semi-supervised general training with test evaluations by Amit Henig and Paz Ilan.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import semisup
from datetime import datetime
from functools import partial
from importlib import import_module

import os
import numpy as np
import time


def getDefaultParams():
    """
    This function builds a set of default parameters - the ones for the article for mnist without our additions.
    :return: run parameters dictionary
    """
    params = dict()
    params['sup_per_class'] = 10 # How many labeled samples per class to take, -1 means all
    params['sup_seed'] = 5 # Labeled example selection seed
    params['sup_per_batch'] = 10 # Batch size for the labeled set for each class
    params['unsup_batch_size'] = 100 # Batch size of the unlabeled samples
    params['unsup_samples'] = -1 # How many unlabeled samples to take, -1 means all
    params['eval_interval'] = 50 # Model evaluation interval, how many epochs between every data save and print
    params['learning_rate'] = 1e-3 # Learning rate
    params['decay_factor'] = 0.33 # Factor for learning rate decay
    params['decay_steps'] = 5000 # Steps between each decay of learning rate
    params['visit_weight'] = 1.0 # Weight for the visit loss
    params['max_steps'] = 20000 # Total epochs in the training
    params['logdir'] = './log/semisup_mnist' # Path to save the logs to
    params['reg_M_weight'] = 1e-3 # Weight for l2 regularization on Mnet
    # M_depth_struct is a struct holding the structure of Mnet in the run, list of list of layers:
    # [neuron amount,activation,fc/conv (nothing=conv)]
    params['M_depth_struct'] = [[64, "sigmoid"],[32, "sigmoid"],[32, "sigmoid"],[16, "sigmoid"],[16, "sigmoid"]]
    # training_regime is a list of list of lists formatted:
    # [[[vars to train],[losses to use],[num_epochs]]...]
    params['training_regime'] = [[[[["emb", "logit"], ["logit", "semisup"]]], 0]]
    params['run_name'] = '' # Name for the run to be saved under (for logs)
    params['dataset'] = 'mnist' # Dataset name, needs data under data folder (we checked mnist is working and enabled fashion mnist)
    params['architecture'] = 'mnist_model' # Model name to use, just needs to be in a part of the file architecture.py
    params['collect_runtime_data'] = False # Flag whether to collect runtime statistics (graph runtimes for tensorboard)
    params['M_activation'] = 'None' # Which activation to choose for last Mnet layer
    params['M_dropout'] = 0 # Dropout prectange for Mnet
    params['minimum_learning_rate'] = 0 # Lowest allowable learning rate
    params['triplet_margin'] = 1.5 # Margin used for all triplet losses
    params['reg_embedding'] = 1e-3 # L2 Regulariztion weight value for the embedding network
    params['logit_weight'] = 1.0 # Weight for logit loss
    params['Msup_weight'] = 1.0 # Weight for our supervised comparison loss
    params['triplet_hard_mining'] = False # Whether to use hard mining for triplets or not
    params['M_batch_norm'] = True # Whether to use batch normalization in Mnet or not
    params['save_model'] = False # Save model checkpoint or not
    return params


def eval_net(model, test_images, test_labels):
    """
    This function evaluates a given model on a test set.
    :param model: Embedding network
    :param test_images: Test images
    :param test_labels: Test images ground truth labels
    :return: tensorflow summary operation of test error
    """
    test_pred = model.classify(test_images).argmax(-1)
    test_err = (test_labels != test_pred).mean() * 100
    print('Test error: %.2f %%' % test_err)
    test_summary = tf.Summary(
        value=[tf.Summary.Value(
            tag='Test Err', simple_value=test_err)])
    return test_summary


def runM(params):
    """
    Main training and evaluation function for our project. 
    Gets a dictionary of params as elaborated above in getDefaultParams(). 
    :param params: params dictionary.
    """
    # Load images, get embedding model and prepare for training
    dataset_tools = import_module('tools.' + params['dataset'])
    architecture = getattr(semisup.architectures, params['architecture'])
    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE
    train_images, train_labels = dataset_tools.get_data('train')
    test_images, test_labels = dataset_tools.get_data('test')
    model_function = partial(architecture,l2_weight = params["reg_embedding"])

    # Default run_name with runtime date if no name given
    if params['run_name'] is '':
      curr_run_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
      curr_run_name = params['run_name']

    # If logdir exists - creates a new one with _1 or _2 if 1 exists and so on...
    currLogDir = params['logdir'] + '/' + curr_run_name
    if os.path.isdir(currLogDir):
      num = 2
      currLogDir_temp = currLogDir + '_' + str(num)
      while os.path.isdir(currLogDir_temp):
          num += 1
          currLogDir_temp = currLogDir + '_' + str(num)
      currLogDir = currLogDir_temp

    # Sample labeled training subset
    seed = params['sup_seed'] if params['sup_seed'] != -1 else None
    sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                           params['sup_per_class'], num_labels, seed)

    # If we don't use all unlabeled data - sample unlabeled data as well
    if params['unsup_samples'] > -1:
        num_unlabeled = len(train_images)
        assert params['unsup_samples'] <= num_unlabeled, (
            'Chose more unlabeled samples ({})'
            ' than there are in the '
            'unlabeled batch ({}).'.format(params['unsup_samples'], num_unlabeled))
        rng = np.random.RandomState(seed=seed)
        train_images = train_images[rng.choice(num_unlabeled, params['unsup_samples'], False)]

    # Create graph
    graph = tf.Graph()
    with graph.as_default():
      # Set up current semisup model to evaluate according to params
      zsup_size = num_labels * params['sup_per_batch']
      zunsup_size = params['unsup_batch_size']
      model = semisup.SemisupModel(model_function, num_labels, image_shape,
                                   zsup_size, zunsup_size, params['M_depth_struct'],
                                   params['reg_M_weight'], params['M_activation'], params['M_dropout']
                                   ,params['triplet_margin'],params['triplet_hard_mining'],params['M_batch_norm'])

      # Set up inputs
      t_unsup_images, t_unsup_labels = semisup.create_input(train_images, train_labels, params['unsup_batch_size'])
      t_sup_images, t_sup_labels = semisup.create_per_class_inputs(sup_by_label, params['sup_per_batch'])
      t_unsup_labels = tf.cast(t_unsup_labels, tf.int32)

      # Compute embeddings, logits and similarities from Mnet
      t_sup_emb = model.image_to_embedding(t_sup_images)
      t_unsup_emb = model.image_to_embedding(t_unsup_images)
      t_sup_logit = model.embedding_to_logit(t_sup_emb)
      t_match_abM = model.embedding_to_M(t_sup_emb, t_unsup_emb)
      t_match_aaM = model.embedding_to_M(t_sup_emb, t_sup_emb)
      t_match_aa = tf.matmul(t_sup_emb, t_sup_emb, transpose_b=True, name='match_aa')
      t_match_bbM = model.embedding_to_M(t_unsup_emb, t_unsup_emb)
      t_match_bb = tf.matmul(t_unsup_emb, t_unsup_emb, transpose_b=True, name='match_bb')

      # Add losses
      # original losses
      model.add_semisup_loss(t_sup_emb, t_unsup_emb, t_match_abM, t_sup_labels, t_unsup_labels, visit_weight=params['visit_weight'])
      model.add_logit_loss(t_sup_logit, t_sup_labels,weight = params['logit_weight'])
      # our losses
      model.add_M_sup_loss(t_match_aaM, t_sup_labels,params['Msup_weight'])
      model.get_sup_loss(t_match_aa, t_sup_labels, '_aa') #mat mul sup loss
      model.get_sup_loss(t_match_bbM, t_unsup_labels, '_bbM')
      model.get_sup_loss(t_match_bb, t_unsup_labels, '_bb')

      t_learning_rate = tf.maximum(
          tf.train.exponential_decay(
              params['learning_rate'],
              model.step,
              params['decay_steps'],
              params['decay_factor'],
              staircase=True),
          params['minimum_learning_rate'])

      # Create training operations according to given training regime
      model.create_train_op(t_learning_rate, params['training_regime'])

      # Collect summaries for tensorboard
      summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(currLogDir, graph)

      # Create checkpoint saver
      saver = tf.train.Saver()

    # Start tensorflow running session
    with tf.Session(graph=graph) as sess:
      # Initializing
      tf.global_variables_initializer().run()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      prev_train_op_idx = 0
      test_summary = eval_net(model, test_images, test_labels)
      summary_writer.add_summary(test_summary, -1)
      if params['collect_runtime_data']:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
      time_per_step = 0

      # Training loop
      for step in range(params['max_steps']):
        startTime = time.time()

        # Get current training operation
        trOp = [model.get_train_op_list(), summary_op]
        curr_train_op_idx = model.train_op_idx

        if params['collect_runtime_data']:
          _, summaries = sess.run(trOp,options=run_options,run_metadata=run_metadata)
        else:
          _, summaries = sess.run(trOp)

        if step > 1:
            time_per_step = ((step - 1) * time_per_step + (time.time() - startTime)) / step
        else:
            time_per_step = time.time() - startTime

        # If its time to evaluate, evaluate! (Also when training op changes)
        if (step + 1) % params['eval_interval'] == 0 or step == 99 or prev_train_op_idx != curr_train_op_idx:
          print(params['run_name'] + ', Step: ' + str(step) + ' Training op: ' + str(curr_train_op_idx))
          print('Step ' + str(step) + '/' + str(params['max_steps']) + ', estimated time remaining: ' + str(round(time_per_step * (params['max_steps'] - step) / 60)) + 'mins')
          test_summary = eval_net(model, test_images, test_labels)
          if params['collect_runtime_data']:
              summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
          summary_writer.add_summary(summaries, step)
          summary_writer.add_summary(test_summary, step)
          print()
          if params['save_model']:
              saver.save(sess, currLogDir, model.step)

        prev_train_op_idx = curr_train_op_idx
      coord.request_stop()
      coord.join(threads)
