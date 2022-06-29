# Copyright (c) 2021, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
r"""Model distillation training binary using a custom training loop.

Distillation attempts to train a smaller student model that mimics the larger
teacher model.

Currently only transformer_learn_values_distill config is
supported for model training.

Example usage:

CONFIG="//learning/genomics/deepconsensus/models/model_configs.py:transformer_learn_values_distill+ccs"
TEACHER_MODEL_DIR=""
OUT_DIR=/tmp

time blaze run -c opt \
//learning/genomics/deepconsensus/models:model_distillation -- \
  --teacher_model_dir ${TEACHER_MODEL_DIR} \
  --params ${CONFIG} \
  --out_dir ${OUT_DIR} \
  --alsologtostderr
"""

import logging
import os
import random
from typing import Optional, Dict

from absl import app
from absl import flags
import ml_collections
from ml_collections.config_flags import config_flags
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import losses_and_metrics
from deepconsensus.models import model_utils


# pylint: disable=unused-import

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('params', None, 'Training configuration.')
flags.DEFINE_string('teacher_model_dir', None,
                    'Path to the teacher model checkpoint.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and model checkpoints.')
flags.DEFINE_string(
    'tpu', None, 'Name of the TPU to use. This gets '
    'populated automatically when using XManager.')
flags.DEFINE_string('tpu_topology', None, 'Tpu topology.')
flags.DEFINE_bool('debug', False,
                  'Enables dumping debug info for TensorBoard Debugger V2.')
flags.DEFINE_bool(
    'write_checkpoint_metrics', False,
    'Whether to write eval metrics for each checkpoint during training.')
flags.DEFINE_bool(
    'eval_and_log_every_step', False, 'Eval and log after every step. '
    'Use this e.g. for testing training and inspecting metrics locally.')


def init_student_from_teacher(
    student_model: tf.keras.Model, teacher_model: tf.keras.Model,
    params: ml_collections.ConfigDict) -> tf.keras.Model:
  """Initialize student model using teacher model weights based on params."""
  row_size = data_providers.get_total_rows(params.max_passes)
  input_shape = (1, row_size, params.max_length, params.num_channels)
  model_utils.print_model_summary(teacher_model, input_shape)
  if params.init_encoder_stack:
    teacher2student_encoder_map = dict(
        zip(params.teacher_encoder_layers, params.student_encoder_layers))

    for teacher_layer_id in teacher2student_encoder_map:
      student_layer_id = teacher2student_encoder_map[teacher_layer_id]
      # Copy attention layer.
      teacher_weights = teacher_model.encoder_stack.layers[teacher_layer_id][
          0].layer.get_weights()
      student_model.encoder_stack.layers[student_layer_id][0].layer.set_weights(
          teacher_weights)
      # Copy ffn layer.
      teacher_weights = teacher_model.encoder_stack.layers[teacher_layer_id][
          1].layer.get_weights()
      student_model.encoder_stack.layers[student_layer_id][1].layer.set_weights(
          teacher_weights)

  if params.init_nonencoder_layers:
    # Get layers with weights that are not in the encoder stack.
    layer_ind_for_copy = []
    for ind, layer in enumerate(teacher_model.layers):
      if (layer.trainable_weights) and ('encoder_stack' not in layer.name):
        layer_ind_for_copy.append(ind)

    for layer_ind in layer_ind_for_copy:
      teacher_weights = teacher_model.get_layer(index=layer_ind).get_weights()
      student_model.get_layer(index=layer_ind).set_weights(teacher_weights)
  return student_model


def get_teacher_model(checkpoint_path: str,
                      strategy: tf.distribute.Strategy) -> tf.keras.Model:
  """Get teacher model with an existing checkpoint."""
  params = model_utils.read_params_from_json(checkpoint_path=checkpoint_path)
  with strategy.scope():
    logging.info('Using checkpoint: %s.', checkpoint_path)
    model = model_utils.get_model(params)
    checkpoint = tf.train.Checkpoint(model=model)
    # Note that the `print_model_summary` is necessary because we need to run a
    # forward pass with the model in order for assert_existing_objects_matched
    # to work as expected.
    # If you don't do this, then  assert_existing_objects_matched will not
    # raise an error even if the wrong checkpoint is used.
    # Some context here: b/148023980.
    row_size = data_providers.get_total_rows(params.max_passes)
    input_shape = (1, row_size, params.max_length, params.num_channels)
    model_utils.print_model_summary(model, input_shape)
    checkpoint.restore(
        checkpoint_path).expect_partial().assert_existing_objects_matched()
  return model


def train_model(teacher_model: tf.keras.Model, out_dir: str,
                params: ml_collections.ConfigDict,
                strategy: tf.distribute.Strategy,
                write_checkpoint_metrics: bool) -> None:
  """Trains the model under the given strategy and params."""
  # Freeze config dict here to ensure it is hashable.
  params = ml_collections.FrozenConfigDict(params)
  model_utils.save_params_as_json(out_dir, params)
  train_dataset, eval_dataset = model_utils.get_datasets(params, strategy)
  steps_per_epoch, steps_per_eval = model_utils.get_step_counts(
      params, FLAGS.eval_and_log_every_step)

  with strategy.scope():
    logging.info('Building model.')
    model = model_utils.get_model(params)
    # Note that the `print_model_summary` is necessary because we need to run a
    # forward pass with the model to be able to initialize student from teacher.
    row_size = data_providers.get_total_rows(params.max_passes)
    input_shape = (1, row_size, params.max_length, params.num_channels)
    model_utils.print_model_summary(model, input_shape)
    logging.info('Done building model.')
    # Initialize student model from teacher based on model params.
    model = init_student_from_teacher(model, teacher_model, params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    train_loss = tf.keras.metrics.Mean(name='loss')
    train_metrics = model_utils.get_deepconsensus_metrics(name_prefix='')
    eval_loss = tf.keras.metrics.Mean(name='loss')
    eval_metrics = model_utils.get_deepconsensus_metrics(name_prefix='')
    student_loss_object = model_utils.get_deepconsensus_loss(
        params, reduction=tf.keras.losses.Reduction.NONE)
    distillation_loss_object = losses_and_metrics.DistillationLoss(
        temperature=params.temperature,
        reduction=tf.keras.losses.Reduction.NONE)

    def compute_all_replica_loss(
        per_example_loss: tf.keras.losses.Loss,
        params: ml_collections.ConfigDict) -> tf.Tensor:
      # We divide per-replica losses by global batch size and sum this value
      # across all replicas to compute average loss scaled by global batch size.
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=params.batch_size)

    def compute_loss(labels: tf.Tensor, student_preds: tf.Tensor,
                     student_logits: tf.Tensor,
                     teacher_logits: tf.Tensor) -> Dict[str, tf.Tensor]:
      per_example_student_loss = student_loss_object(labels, student_preds)
      per_example_distill_loss = distillation_loss_object(
          teacher_logits, student_logits)
      per_example_loss = (
          params.student_alpha * per_example_student_loss +
          params.distill_alpha * per_example_distill_loss)

      losses_dict = {}
      losses_dict['total_loss'] = compute_all_replica_loss(
          per_example_loss, params)
      losses_dict['student_loss'] = compute_all_replica_loss(
          per_example_student_loss, params)
      losses_dict['distill_loss'] = compute_all_replica_loss(
          per_example_distill_loss, params)
      return losses_dict

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    checkpoint, initial_epoch = model_utils.get_checkpoint_and_initial_epoch(
        model, optimizer, out_dir, steps_per_epoch)  # pytype: disable=wrong-arg-types  # typed-keras

  # Create summary writers
  train_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'train'))
  eval_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'eval'))

  def train_step(inputs):
    """Training StepFn."""
    features, labels = inputs
    # Get logits from the teacher model.
    teacher_logits = teacher_model.get_logits(features, training=False)

    with tf.GradientTape() as tape:
      student_logits = model.get_logits(features, training=True)
      student_preds = tf.nn.softmax(student_logits)
      train_losses_dict = compute_loss(labels, student_preds, student_logits,
                                       teacher_logits)
      loss = train_losses_dict['total_loss']
    # Compute gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    # Update weights.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(loss)
    for metric in train_metrics:
      metric.update_state(labels, student_preds)
    return train_losses_dict

  def eval_step(inputs):
    """Eval StepFn."""
    features, labels = inputs
    # Get logits from the teacher model.
    teacher_logits = teacher_model.get_logits(features, training=False)

    student_logits = model.get_logits(features, training=False)
    student_preds = tf.nn.softmax(student_logits)
    eval_losses_dict = compute_loss(labels, student_preds, student_logits,
                                    teacher_logits)

    eval_loss.update_state(eval_losses_dict['total_loss'])
    for metric in eval_metrics:
      metric.update_state(labels, student_preds)
    return eval_losses_dict

  @tf.function
  def distributed_train_step(iterator):
    per_replica_losses_dict = strategy.run(train_step, args=(next(iterator),))
    reduced_train_losses_dict = {}
    for loss_name in per_replica_losses_dict.keys():
      reduced_train_losses_dict[loss_name] = strategy.reduce(
          tf.distribute.ReduceOp.SUM,
          per_replica_losses_dict[loss_name],
          axis=None)
    return reduced_train_losses_dict

  @tf.function
  def distributed_eval_step(iterator):
    per_replica_losses_dict = strategy.run(eval_step, args=(next(iterator),))
    reduced_eval_losses_dict = {}
    for loss_name in per_replica_losses_dict.keys():
      reduced_eval_losses_dict[loss_name] = strategy.reduce(
          tf.distribute.ReduceOp.SUM,
          per_replica_losses_dict[loss_name],
          axis=None)
    return reduced_eval_losses_dict

  log_steps = 100
  if FLAGS.eval_and_log_every_step:
    log_steps = 1
  train_iterator = iter(train_dataset)
  eval_iterator = iter(eval_dataset)
  min_eval_loss = 1e6
  for epoch in range(initial_epoch, params['num_epochs']):
    logging.info('Starting to run epoch: %s', epoch)
    with train_writer.as_default():
      for step in range(steps_per_epoch):
        reduced_train_losses = distributed_train_step(train_iterator)
        if step % log_steps == 0:
          model_utils.log_and_save_metrics(
              epoch=epoch,
              step=step,
              total_steps=steps_per_epoch,
              optimizer=optimizer,
              losses_dict=reduced_train_losses,
              metrics=train_metrics,
              training=True)
    with eval_writer.as_default():
      for step in range(steps_per_eval):
        reduced_eval_losses = distributed_eval_step(eval_iterator)
      model_utils.log_and_save_metrics(
          epoch=epoch,
          step=step,
          total_steps=steps_per_eval,
          optimizer=optimizer,
          losses_dict=reduced_eval_losses,
          metrics=eval_metrics,
          training=False)
    checkpoint_name = model_utils.save_checkpoint(checkpoint, out_dir,
                                                  train_metrics, eval_metrics,
                                                  write_checkpoint_metrics)
    if min_eval_loss > float(eval_loss.result()):
      min_eval_loss = float(eval_loss.result())
      with tf.io.gfile.GFile(os.path.join(out_dir, 'best_checkpoint.txt'),
                             'w') as f:
        f.write(os.path.basename(checkpoint_name))
    model_utils.reset_all_metrics([train_loss, eval_loss] + train_metrics +
                                  eval_metrics)


def train(teacher_model_dir: str,
          out_dir: str,
          params: ml_collections.ConfigDict,
          tpu: Optional[str],
          tpu_topology: Optional[str],
          write_checkpoint_metrics: bool,
          debug: Optional[bool] = False):
  """Run the model training and return evaluation output."""
  model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
  random.seed(params.seed)
  tf.random.set_seed(params.seed)
  os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'False'
  while True:
    try:
      if tpu is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
      elif debug:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
      else:
        strategy = tf.distribute.MirroredStrategy()
      teacher_model = get_teacher_model(teacher_model_dir, strategy=strategy)
      train_model(teacher_model, out_dir, params, strategy,
                  write_checkpoint_metrics)
      break
    except tf.errors.UnavailableError:
      continue


def main(unused_args=None):
  train(FLAGS.teacher_model_dir, FLAGS.out_dir, FLAGS.params, FLAGS.tpu,
        FLAGS.tpu_topology, FLAGS.write_checkpoint_metrics, FLAGS.debug)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'teacher_model_dir',
      'params',
      'out_dir',
  ])
  app.run(main)
