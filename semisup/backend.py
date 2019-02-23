"""
- Revised version of backend.py from https://github.com/haeusser/learning_by_association for learning to learn by association
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

  Utility functions for Association-based semisupervised training.

- Revision by Amit Henig and Paz Ilan.
- NOTES by Amit Henig and Paz Ilan within code to mark changes made.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops

# NOTE: Unchanged function from [1]
def create_input(input_images, input_labels, batch_size):
  """Create preloaded data batch inputs.

  Args:
    input_images: 4D numpy array of input images.
    input_labels: 2D numpy array of labels.
    batch_size: Size of batches that will be produced.

  Returns:
    A list containing the images and labels batches.
  """
  if input_labels is not None:
    image, label = tf.train.slice_input_producer([input_images, input_labels])
    return tf.train.batch([image, label], batch_size=batch_size)
  else:
    image = tf.train.slice_input_producer([input_images])
    return tf.train.batch(image, batch_size=batch_size)

# NOTE: Unchanged function from [1]
def create_per_class_inputs(image_by_class, n_per_class, class_labels=None):
  """Create batch inputs with specified number of samples per class.

  Args:
    image_by_class: List of image arrays, where image_by_class[i] containts
        images sampled from the class class_labels[i].
    n_per_class: Number of samples per class in the output batch.
    class_labels: List of class labels. Equals to range(len(image_by_class)) if
        not provided.

  Returns:
    images: Tensor of n_per_class*len(image_by_class) images.
    labels: Tensor of same number of labels.
  """
  if class_labels is None:
    class_labels = np.arange(len(image_by_class))
  batch_images, batch_labels = [], []
  for images, label in zip(image_by_class, class_labels):
    labels = tf.fill([len(images)], label)
    images, labels = create_input(images, labels, n_per_class)
    batch_images.append(images)
    batch_labels.append(labels)
  return tf.concat(batch_images, 0), tf.concat(batch_labels, 0)

# NOTE: Unchanged function from [1]
def sample_by_label(images, labels, n_per_label, num_labels, seed=None):
  """Extract equal number of sampels per class."""
  res = []
  rng = np.random.RandomState(seed=seed)
  for i in range(num_labels):
    a = images[labels == i]
    if n_per_label == -1:  # use all available labeled data
      res.append(a)
    else:  # use randomly chosen subset
      res.append(a[rng.choice(len(a), n_per_label, False)])
  return res

# NOTE: Unchanged function from [1]
def create_virt_emb(n, size):
  """Create virtual embeddings."""
  emb = slim.variables.model_variable(
      name='virt_emb',
      shape=[n, size],
      dtype=tf.float32,
      trainable=True,
      initializer=tf.random_normal_initializer(stddev=0.01))
  return emb

# NOTE: Unchanged function from [1]
def confusion_matrix(labels, predictions, num_labels):
  """Compute the confusion matrix."""
  rows = []
  for i in range(num_labels):
    row = np.bincount(predictions[labels == i], minlength=num_labels)
    rows.append(row)
  return np.vstack(rows)

# NOTE: Changed functions within model from [1]
class SemisupModel(object):
  """Helper class for setting up semi-supervised training."""

  # NOTE: Changed function from [1]
  def __init__(self, model_func, num_labels, input_shape, sup_Size, unsup_Size, M_depth_struct = [],
               reg_M_weight = 0.0, M_activation = 'sigmoid', M_dropout = 0.0, triplet_margin=0.01,
               triplet_hard_mining = False,
               M_batch_norm = True,test_in=None):
    """Initialize SemisupModel class.

    Creates an evaluation graph for the provided model_func.

    Args:
      model_func: Model function. It should receive a tensor of images as
          the first argument, along with the 'is_training' flag.
      num_labels: Number of taget classes.
      input_shape: List, containing input images shape in form
          [height, width, channel_num].
      sup_Size: Integer, size of supervised batch.
      unsup_Size: Integer, size of unsupervised batch.
      M_depth_struct = Struct holding the configuration for the Mnet network.
      reg_M_weight = Float, Regularization weight for Mnet. Default 0.0.
      M_activation = String:valid values are 'sigmoid','relu' or 'elu'. Activation function for the last layer of Mnet.
        Default 'sigmoid'.
      M_dropout = Float. Dropout precentage for Mnet (doesnt apply to last layer). Default 0.0.
      triplet_margin = Float. Margin used for all triplet based losses. Default 0.01.
      triplet_hard_mining = Bool. Flag to choose between "hard mining" for triplet based losses or not. Default False.
      M_batch_norm = Bool. Flag to enable batch normalization in Mnet (doesnt apply to last layer).Default True.
      test_in: None or a tensor holding test images. If None, a placeholder will
        be created.
    """
    # NOTE: Unchanged initializations from [1]
    self.num_labels = num_labels
    self.step = slim.get_or_create_global_step()
    self.test_batch_size = 100
    self.model_func = model_func

    if test_in is not None:
      self.test_in = test_in
    else:
      self.test_in = tf.placeholder(np.float32, [None] + input_shape, 'test_in')

    self.test_emb = self.image_to_embedding(self.test_in, is_training=False)
    self.test_logit = self.embedding_to_logit(self.test_emb, is_training=False)

    # NOTE: Added initialization for Mnet
    self.sup_per_batch = sup_Size//num_labels
    self.reg_M_weight = reg_M_weight
    self.M_activation = M_activation
    self.M_dropout = M_dropout
    self.triplet_margin = triplet_margin
    self.triplet_hard_mining = triplet_hard_mining
    self.M_depth = len(M_depth_struct) + 1
    self.M_features = []
    self.M_activations = []
    self.M_use_conv = [True]
    self.M_batch_norm = M_batch_norm
    for layer in M_depth_struct:
      self.M_features += [layer[0]]
      self.M_activations += [layer[1]]
      self.M_use_conv += [len(layer) < 3 or layer[2] == 'conv']
    self.M_features += [1]
    self.M_activations += [M_activation]

    self.tf_activations = {
      'None': None,
      'sigmoid': tf.nn.sigmoid,
      'relu': tf.nn.relu,
      'elu': tf.nn.elu,
    }

    self.end_losses = dict()

    embAShape = (self.test_emb.get_shape().as_list())[-1]
    self.test_embA = tf.placeholder(np.float32, [sup_Size, embAShape], 'test_embA')
    self.test_embB = tf.placeholder(np.float32, [unsup_Size, embAShape], 'test_embA')
    self.test_machine_m = self.embedding_to_M(self.test_embA, self.test_embB, is_training=False)

  # NOTE: Unchanged function from [1]
  def image_to_embedding(self, images, is_training=True):
    """Create a graph, transforming images into embedding vectors."""
    with tf.variable_scope('net', reuse=is_training):
      return self.model_func(images, is_training=is_training)

  # NOTE: Unchanged function from [1]
  def embedding_to_logit(self, embedding, is_training=True):
    """Create a graph, transforming embedding vectors to logit classs scores."""
    with tf.variable_scope('logitNet', reuse=is_training):
      return slim.fully_connected(
          embedding,
          self.num_labels,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(1e-4),
          scope='fc_S')

  # NOTE: New function for Mnet
  def embedding_to_M(self, embA, embB, is_training=True):
    """Create a graph, transforming embedding vectors similarity scores using Mnet.
      Args:
          embA: [N, emb_size] tensor with supervised embedding vectors.
          embB: [M, emb_size] tensor with unsupervised embedding vectors.
          is_training: bool to mark whether training is active currently
    """
    with tf.variable_scope('mnet', reuse=is_training):
      # get initial details
      embAShape = embA.get_shape().as_list()
      embBShape = embB.get_shape().as_list()
      N = embAShape[0]
      M = embBShape[0]
      L = embAShape[1]

      # reshape A,B embedding matrices to fit our Pair-wise fully connected convolutional layer
      embB_flat = tf.reshape(embB, [1, M * L])
      embBRep = tf.tile(embB_flat, [N, 1])
      embARep = tf.tile(embA, [1, M])
      embABRep = tf.stack([embBRep, embARep], axis=2)
      embABRepExp = tf.expand_dims(embABRep, 0)
      output = embABRepExp

      # build Mnet according to chosen structure in params
      for i in range(self.M_depth):
        w = L if i == 0 else 1
        scope = 'layers_M_' + str(i)
        activation = self.tf_activations[self.M_activations[i]]
        norm_fn = None if i == self.M_depth - 1 or not self.M_batch_norm else slim.batch_norm
        norm_params = None if i == self.M_depth - 1 else{'is_training': is_training,'decay': 0.99}

        if self.M_use_conv[i]:
          output = slim.conv2d(output, self.M_features[i], [1, w], stride=[1, w], activation_fn=activation,
                               padding='VALID', weights_regularizer=slim.l2_regularizer(self.reg_M_weight),
                               scope=scope, reuse=is_training,
                               normalizer_fn=norm_fn, normalizer_params=norm_params)
        else:
          output = slim.fully_connected(
            output,
            self.M_features[i],
            activation_fn=activation,
            weights_regularizer = slim.l2_regularizer(self.reg_M_weight),
            scope = scope, reuse = is_training,
            normalizer_fn = norm_fn, normalizer_params = norm_params)
        if i<self.M_depth and self.M_dropout>0:
            output = slim.dropout(output, 1-self.M_dropout, is_training=is_training,
                               scope= 'dropout' + "_" + str(i))

      return tf.squeeze(output)

  # NOTE: New function for Mnet
  def get_sup_loss(self,match,labels,name_ext=''):
    """
    :param match: similarity matrix tensor output from Mnet.
    :param labels: ground truth labels.
    :param name_ext: extension to name for different losses.
    :return: supervised comparison loss for given match and label inputs, as described in our project.
    """
    p = tf.nn.softmax(match, name='p'+name_ext)
    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(
        equality_matrix, [1], keep_dims=True))
    loss_aa = tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p),
        scope='loss'+name_ext)
    tf.summary.scalar('Loss'+name_ext, loss_aa)
    return loss_aa

  # NOTE: New function for Mnet
  def get_triplet_loss(self, p, labels, margin=0.01, name_ext='', hard_mining=False):
    """
    This funtion gets a similarity matrix p, ground truth labels and a margin. It then uses our defined supervised and
    semi supervised triplet loss formulations (differ only in the input matrix p) to calculate semisupervised/supervised
    triplet losses.
    :param p: similarity probability tensor.
    :param labels: ground truth labels.
    :param margin: margin parameter for triplet loss as explained in our project.
    :param name_ext: extension to name for different losses.
    :param hard_mining: whether to calculate the loss for all examples or only the hardest ones.
    :return: triplet loss for given p and label inputs, as described in our project.
    """
    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)

    if not hard_mining:
      pShape = p.get_shape().as_list()
      p_standing = tf.reshape(p,[pShape[0],1,pShape[1]])
      diff_probs = p-p_standing
      eq_standing = tf.reshape(equality_matrix,[pShape[0],1,pShape[1]])
      substraction_sign = equality_matrix - eq_standing #pos-pos = 0, false - pos = 1,pos - false = -1
      trip_loss = tf.reduce_sum(tf.maximum(0.0,(tf.multiply(diff_probs,substraction_sign)+tf.constant(margin)))/
                                tf.reduce_sum(tf.abs(substraction_sign)),name='loss_triplet_hard'+name_ext)
    else:
      pos_scores = tf.reduce_min(p + (1.0 - equality_matrix)*100, [1], keepdims=True)
      false_scores = tf.reduce_max(tf.multiply(p, 1.0 - equality_matrix), [1], keepdims=True)
      loss = tf.maximum(0., false_scores - pos_scores + margin)
      trip_loss = tf.multiply(tf.cast(tf.constant(1), tf.float32), tf.reduce_mean(loss,name='loss_triplet'+name_ext))

    tf.summary.scalar('Loss_triplet'+name_ext, trip_loss)
    return trip_loss

  # NOTE: New function for Mnet
  def get_triplet_loss_unsup(self, p, labels, margin=0.01, name_ext='', hard_mining = False):
    """
      This funtion gets a similarity matrix p, ground truth labels and a margin. It then uses our defined unsupervised
      triplet loss formulation to calculate unsupervised triplet loss.
      :param p: similarity probability tensor.
      :param labels: ground truth labels.
      :param margin: margin parameter for triplet loss as explained in our project.
      :param name_ext: extension to name for different losses.
      :param hard_mining: whether to calculate the loss for all examples or only the hardest ones.
      :return: unsupervised triplet loss for given p and label inputs, as described in our project.
      """
    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(
      equality_matrix, [1], keep_dims=True))
    p_avg_per_class = tf.matmul(p, p_target, transpose_a=True, name='p_per_class_average' + name_ext)

    if hard_mining:
      top_scores, _ = tf.math.top_k(p_avg_per_class, self.sup_per_batch + 1, sorted=True)
      diff_top_scores = tf.matmul(top_scores,
                                  tf.cast(np.concatenate([np.zeros([1,self.sup_per_batch-1]),np.array([[-1,1]])],1),tf.float32),
                                transpose_b=True)
      loss = tf.maximum(0., diff_top_scores + margin)
      trip_loss = tf.reduce_mean(loss,name='loss_triplet2_hard'+name_ext)
    else:
      top_scores, top_indices = tf.math.top_k(p_avg_per_class, self.sup_per_batch, sorted=True)
      correct_class_mat = tf.reduce_sum(tf.one_hot(top_indices, labels.get_shape()[-1]), [1])
      shape = p_avg_per_class.get_shape().as_list()
      p_avg_per_class_standing = tf.reshape(p_avg_per_class, [shape[0], 1, shape[1]])
      diff_avg_probs = p_avg_per_class - p_avg_per_class_standing
      correct_class_mat_standing = tf.reshape(correct_class_mat, [shape[0], 1, shape[1]])
      substraction_sign = correct_class_mat - correct_class_mat_standing  # pos-pos = 0, false - pos = 1,pos - false = -1
      trip_loss = tf.reduce_sum(
        tf.maximum(0.0, (tf.multiply(diff_avg_probs, substraction_sign) + tf.constant(margin))) / tf.reduce_sum(
        tf.abs(substraction_sign)),name='loss_triplet2'+name_ext)

    tf.summary.scalar('Loss_triplet2' + name_ext, trip_loss)
    return trip_loss

  # NOTE: New function for Mnet
  def add_M_sup_loss(self, match_aa, labels, Msup_weight = 1.0):
    """
    This function gets the similarity matrix and ground truth labels and adds our additional supervised
    losses to the model, supervised comparison loss and supervised triplet loss , as described in our project.
    :param match_aa: similarity matrix tensor output from Mnet for the labeled data.
    :param labels: ground truth labels.
    :param Msup_weight: supervised comparison loss weight.
    :return:
    """
    self.end_losses['M_sup'] = Msup_weight*self.get_sup_loss(match_aa,labels,'_aaM')
    p_aaM = tf.nn.sigmoid(match_aa, name='p_aa_trip')
    self.end_losses['M_triplet'] = self.get_triplet_loss(p_aaM,labels,self.triplet_margin,'_aaM',self.triplet_hard_mining)

  # NOTE: Changed function for Mnet
  def add_semisup_loss(self, a, b, matchM_ab, labels, labels_b, walker_weight=1.0, visit_weight=1.0):
    """Add semi-supervised classification loss to the model.

    The loss constist of two terms: "walker" and "visit".

    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      matchM_ab: similarity matrix tensor output from Mnet for the unlabeled and labeled data.
      labels_b: ground truth labels for the unlabeled data - ONLY USED TO CALCULATE LOSS FOR EXPERIMENTS! NOT FOR TRAINING!
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """
    # NOTE: Start of unchange section
    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(
        equality_matrix, [1], keep_dims=True))

    # Naive comparison
    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    self.loss_aba = tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p_aba),
        weights=walker_weight,
        scope='loss_aba')
    self.visit_loss = self.add_visit_loss(p_ab, visit_weight)

    tf.summary.scalar('Loss_aba', self.loss_aba)

    # NOTE: end of unchanged section, start of our additions.

    # added semisupervised and unsupervised triplet losses to [1]'s original similarity scores.
    self.end_losses['semisup_triplet'] = self.get_triplet_loss(p_aba,labels,self.triplet_margin,'_aba',self.triplet_hard_mining)
    self.end_losses['semisup_triplet2_sig'] = self.get_triplet_loss_unsup(tf.nn.sigmoid(match_ab, name='p_sig_ab'),
                                                                      labels, self.triplet_margin, '_sig_ab',
                                                                      self.triplet_hard_mining)

    # added loss on transition probability matrix just for examination, does not teach the network anything!
    equality_matrix_ab = tf.equal(tf.reshape(labels, [-1, 1]), labels_b)
    equality_matrix_ab = tf.cast(equality_matrix_ab, tf.float32)
    p_ab_target = (equality_matrix_ab / tf.reduce_sum(
      equality_matrix_ab, [1], keep_dims=True))
    loss_ab = tf.losses.softmax_cross_entropy(
        p_ab_target,
        tf.log(1e-8 + p_ab),
        weights=walker_weight,
        scope='loss_ab')
    tf.summary.scalar('Loss_ab', loss_ab)


    # Mnet similarity matrix to walk probabilities
    p_abM = tf.nn.softmax(matchM_ab, name='p_abM')
    p_baM = tf.nn.softmax(tf.transpose(matchM_ab), name='p_baM')
    p_abaM = tf.matmul(p_abM, p_baM, name='p_abaM')

    # walker and visit loss for Mnet
    self.loss_abaM = tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p_abaM),
        weights=walker_weight,
        scope='loss_abaM')
    self.visit_lossM = self.add_visit_loss(p_abM, visit_weight, 'M')
    tf.summary.scalar('Loss_abaM', self.loss_abaM)

    # semisupervised and unsupervised triplet losses for Mnet as described in our project.
    self.end_losses['Msemisup_triplet'] = self.get_triplet_loss(p_abaM, labels, self.triplet_margin, '_abaM',
                                                                self.triplet_hard_mining)
    self.end_losses['Msemisup_triplet2_sig'] = self.get_triplet_loss_unsup(tf.nn.sigmoid(matchM_ab, name='p_sig_abM'),
                                                                      labels, self.triplet_margin, '_sig_abM',
                                                                      self.triplet_hard_mining)

    # added loss on transition probability matrix just for examination, does not teach the network anything!
    loss_abM = tf.losses.softmax_cross_entropy(
      p_ab_target,
      tf.log(1e-8 + p_abM),
      weights=walker_weight,
      scope='loss_abM')
    tf.summary.scalar('Loss_abM', loss_abM)

  # NOTE: Unchanged function from [1]
  def add_visit_loss(self, p, weight=1.0, name_ext = ''):
    """Add the "visit" loss to the model.

    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(
        p, [0], keep_dims=True, name='visit_prob' + name_ext)
    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / (1e-18 + tf.cast(t_nb, tf.float32))),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit' + name_ext)

    tf.summary.scalar('Loss_Visit' + name_ext, visit_loss)
    return visit_loss

  # NOTE: Unchanged function from [1]
  def add_logit_loss(self, logits, labels, weight=1.0, smoothing=0.0):
    """Add supervised classification loss to the model."""

    self.end_losses['logit'] = tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, logits.get_shape()[-1]),
        logits,
        scope='loss_logit',
        weights=weight,
        label_smoothing=smoothing)

    tf.summary.scalar('Loss_Logit', self.end_losses['logit'])

  # NOTE: Changed function drasticaly to enable training regiments.
  def create_train_op(self, learning_rate, training_reg):
    """
    Creates all training operations according to schedule in params.
    :param learning_rate: learning rate for training.
    :param training_reg: training regime. List of list of lists formatted [[[vars to train],[losses to use],[num_epochs]]...]
    """

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    #NOTE: all changed in the function from here onward.
    # add original semisupervised losses for both Mnet and [1]
    self.end_losses['semisup'] = self.loss_aba+self.visit_loss
    self.end_losses['M'] = self.loss_abaM + self.visit_lossM

    # get all regularization losses
    reg_loss = dict()
    reg_loss['emb'] = tf.losses.get_regularization_losses(scope='net')
    reg_loss['logit'] = tf.losses.get_regularization_losses(scope='logitNet')
    reg_loss['M'] = tf.losses.get_regularization_losses(scope='mnet')

    # collect losses for analysis
    tf.summary.scalar('reg_loss_emb',  math_ops.add_n(reg_loss['emb']))
    tf.summary.scalar('reg_loss_logit',  math_ops.add_n(reg_loss['logit']))
    tf.summary.scalar('reg_loss_M',  math_ops.add_n(reg_loss['M']))
    tf.summary.scalar('Learning_Rate', learning_rate)
    tf.summary.scalar('Loss_TotalOrig', math_ops.add_n(reg_loss['emb'] + reg_loss['logit'] + [self.end_losses['logit']]
                                                       + [self.end_losses['semisup']]))

    # create training variable dictionary
    train_vars = dict()
    train_vars['emb'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "net")
    train_vars['logit'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "logitNet")
    train_vars['M'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "mnet")

    # create trainer
    trainer = tf.train.AdamOptimizer(learning_rate)

    # go over all the training schedule and collect the appropriate variable and losses for each to create
    # the appropriate training op
    #[[[[vars],[losses],duration],....]]]
    self.train_op_idx = 0
    self.train_op_counter = 0
    self.train_op = []
    for train_period in training_reg:
      train_regis_list = train_period[0]
      reg_duration = train_period[1]
      op_list = []
      training_reg_number = 0
      for tr_reg in train_regis_list:
        training_reg_number += 1
        train_vars_reg = []
        total_loss_reg_list = []
        for var_group in tr_reg[0]:
          train_vars_reg += train_vars[var_group]
          total_loss_reg_list += reg_loss[var_group]
        for loss_group in tr_reg[1]:
          total_loss_reg_list += [self.end_losses[loss_group]]
        total_loss_reg = math_ops.add_n(total_loss_reg_list)
        op_list += [slim.learning.create_train_op(total_loss_reg, trainer, variables_to_train = train_vars_reg)]
        tf.summary.scalar('Loss_Total_'+str(training_reg_number), total_loss_reg)
      self.train_op.append([op_list, reg_duration])

  # NOTE: New function to enable training regiments.
  def get_train_op_list(self):
    """
    This function uses the current step saved in the model to get the correct training operation create in
    create_training_op according to params.
    :return: Current step's training operation
    """
    # assumes function is called between epochs
    return_op = self.train_op[self.train_op_idx][0]
    self.train_op_counter += 1

    # check training op duration expiration
    if self.train_op[self.train_op_idx][1] != 0 and self.train_op_counter == self.train_op[self.train_op_idx][1]:
      self.train_op_counter = 0
      self.train_op_idx = (self.train_op_idx+1)%len(self.train_op)

    return return_op

  # NOTE: Unchanged function from [1]
  def calc_embedding(self, images, endpoint):
    """Evaluate 'endpoint' tensor for all 'images' using batches."""
    batch_size = self.test_batch_size
    emb = []
    for i in range(0, len(images), batch_size):
      emb.append(endpoint.eval({self.test_in: images[i:i + batch_size]}))
    return np.concatenate(emb)

  # NOTE: Unchanged function from [1]
  def calc_machine_output(self, embeddings):
    """Evaluate 'endpoint' tensor for all 'images' using batches."""
    output = self.test_machine_m.eval({self.test_embA: embeddings,self.test_embB: embeddings})
    return output

  # NOTE: Unchanged function from [1]
  def classify(self, images):
    """Compute logit scores for provided images."""
    return self.calc_embedding(images, self.test_logit)
