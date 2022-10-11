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
  --xm_runlocal \
  --alsologtostderr
"""

import datetime
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
from deepconsensus.utils import dc_constants


# pylint: disable=unused-import

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('params', None, 'Training configuration.')
_TEACHER_MODEL_DIR = flags.DEFINE_string(
    'teacher_model_dir', None, 'Path to the teacher model checkpoint.')
_OUT_DIR = flags.DEFINE_string('out_dir', None,
                               'Output path for logs and model checkpoints.')
_TPU = flags.DEFINE_string(
    'tpu', None, 'Name of the TPU to use. This gets '
    'populated automatically when using XManager.')
_TPU_TOPOLOGY = flags.DEFINE_string('tpu_topology', None, 'Tpu topology.')
_DEBUG = flags.DEFINE_bool(
    'debug', False, 'Enables dumping debug info for TensorBoard Debugger V2.')
_WRITE_CHECKPOINT_METRICS = flags.DEFINE_bool(
    'write_checkpoint_metrics', False,
    'Whether to write eval metrics for each checkpoint during training.')
_EVAL_AND_LOG_EVERY_STEP = flags.DEFINE_bool(
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
      params, _EVAL_AND_LOG_EVERY_STEP.value)
  # Number of steps this model will train for.
  total_train_steps = steps_per_epoch * params['num_epochs']
  logging.info('Total training steps = %s', total_train_steps)

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
    epoch_checkpoint = os.path.join(out_dir, 'epoch_checkpoint.txt')
    model = init_student_from_teacher(model, teacher_model, params)

    # Calculate the number of steps to decay the learning rate over.
    # Usually this number is the total training steps. However, since we train
    # the model for more epochs to obtain the final model, decay_steps is based
    # on the total training steps taken during final training.
    decay_steps = steps_per_epoch * params['num_epochs_for_decay']
    optimizer = model_utils.create_optimizer(params, decay_steps)

    train_loss = tf.keras.metrics.Mean(name='train/loss')
    train_metrics = model_utils.get_deepconsensus_metrics(name_prefix='train/')
    eval_loss = tf.keras.metrics.Mean(name='eval/loss')
    eval_metrics = model_utils.get_deepconsensus_metrics(name_prefix='eval/')
    # Create an alignment metric object that will be used in yield calculation.
    alignment_metric_yield_obj = losses_and_metrics.AlignmentMetric(
        name='alignment_metric_yield')
    # Create loss objects.
    student_loss_object = model_utils.get_deepconsensus_loss(
        params, reduction=tf.keras.losses.Reduction.NONE)
    distillation_loss_object = losses_and_metrics.DistillationLoss(
        temperature=params.temperature,
        logit_loss=tf.keras.losses.get(params.logit_loss_identifier),
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
        model, optimizer, epoch_checkpoint)  # pytype: disable=wrong-arg-types  # typed-keras

  # Create summary writers
  train_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'train'))
  eval_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'eval'))

  def train_step(inputs):
    """Training StepFn."""
    features, labels = inputs
    # Get logits from the teacher model.
    teacher_intermediate_outputs_dict = teacher_model.get_intermediate_outputs(
        features, training=False)
    teacher_logits = teacher_intermediate_outputs_dict['logits']

    with tf.GradientTape() as tape:
      student_intermediate_outputs_dict = model.get_intermediate_outputs(
          features, training=True)
      student_logits = student_intermediate_outputs_dict['logits']
      student_preds = tf.nn.softmax(student_logits)
      train_losses_dict = compute_loss(labels, student_preds, student_logits,
                                       teacher_logits)
      loss = train_losses_dict['total_loss']
    # Compute gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    # Update weights.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(loss)

    # Calculate identity for CCS and the DC prediction.
    ccs = model_utils.get_ccs_from_example(features, params)
    (identity_ccs,
     identity_pred) = losses_and_metrics.get_batch_identity_ccs_pred(
         ccs, student_preds, labels, alignment_metric_yield_obj)
    # Update metrics.
    model_utils.update_metrics(train_metrics, labels, student_preds,
                               identity_pred, identity_ccs)
    return train_losses_dict

  def eval_step(inputs):
    """Eval StepFn."""
    features, labels = inputs
    # Get logits from the teacher model.
    teacher_intermediate_outputs_dict = teacher_model.get_intermediate_outputs(
        features, training=False)
    teacher_logits = teacher_intermediate_outputs_dict['logits']

    student_intermediate_outputs_dict = model.get_intermediate_outputs(
        features, training=False)
    student_logits = student_intermediate_outputs_dict['logits']
    student_preds = tf.nn.softmax(student_logits)
    eval_losses_dict = compute_loss(labels, student_preds, student_logits,
                                    teacher_logits)

    eval_loss.update_state(eval_losses_dict['total_loss'])
    # Calculate identity for CCS and the DC prediction.
    ccs = model_utils.get_ccs_from_example(features, params)
    (identity_ccs,
     identity_pred) = losses_and_metrics.get_batch_identity_ccs_pred(
         ccs, student_preds, labels, alignment_metric_yield_obj)
    # Update metrics.
    model_utils.update_metrics(eval_metrics, labels, student_preds,
                               identity_pred, identity_ccs)
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

  log_train_steps = 100
  log_eval_steps = 3000
  if _EVAL_AND_LOG_EVERY_STEP.value:
    log_train_steps = 1
  train_iterator = iter(train_dataset)
  eval_iterator = iter(eval_dataset)

  # Decide the best checkpoiht using main eval metric.
  max_main_eval_metric = 0.0
  # From a list of eval metrics get the main eval metric.
  main_eval_metric = next(
      (metric for metric in eval_metrics
       if metric.name == dc_constants.MAIN_EVAL_METRIC_NAME), None)
  if not main_eval_metric:
    raise ValueError('No eval metric found.')

  for epoch in range(initial_epoch, params['num_epochs']):
    logging.info('Starting to run epoch: %s', epoch)
    train_time_start = datetime.datetime.now()
    for step_train in range(steps_per_epoch):
      distributed_train_step(train_iterator)
      # Log and reset train metrics.
      if optimizer.iterations % log_train_steps == 0:
        train_time_end = datetime.datetime.now()
        train_steps_per_second = log_train_steps / (
            train_time_end - train_time_start).total_seconds()
        with train_writer.as_default():
          model_utils.log_and_save_metrics(
              epoch=epoch,
              num_epochs=params['num_epochs'],
              step=step_train,
              total_steps=steps_per_epoch,
              optimizer=optimizer,
              metrics=[train_loss] + train_metrics,
              training=True,
              steps_per_second=train_steps_per_second)
          train_time_start = datetime.datetime.now()
      # Log eval metrics, save checkpoint, and reset eval metrics every
      # log_eval_steps and at the end of training.
      if (optimizer.iterations % log_eval_steps == 0) or (optimizer.iterations
                                                          == total_train_steps):
        # Run evalution on the whole eval dataset and collect metrics.
        eval_time_start = datetime.datetime.now()
        for step_eval in range(steps_per_eval):
          distributed_eval_step(eval_iterator)
        eval_time_end = datetime.datetime.now()
        eval_steps_per_second = steps_per_eval / (
            eval_time_end - eval_time_start).total_seconds()
        # Save checkpoint.
        checkpoint_name = model_utils.save_checkpoint(
            checkpoint, out_dir, [eval_loss] + eval_metrics,
            write_checkpoint_metrics)
        # Record the best checkpoint based on the main eval metric.
        main_eval_metric_val = float(main_eval_metric.result())
        if main_eval_metric_val >= max_main_eval_metric:
          max_main_eval_metric = main_eval_metric_val
          with tf.io.gfile.GFile(
              os.path.join(out_dir, 'best_checkpoint.txt'), 'w') as f:
            f.write(os.path.basename(checkpoint_name))
        # Log metrics on the eval set, this must be done at the end since
        # log_and_save_metrics will reset the eval metrics values.
        with eval_writer.as_default():
          model_utils.log_and_save_metrics(
              epoch=epoch,
              num_epochs=params['num_epochs'],
              step=step_eval,
              total_steps=steps_per_eval,
              optimizer=optimizer,
              metrics=[eval_loss] + eval_metrics,
              training=False,
              steps_per_second=eval_steps_per_second)
        # Reset timer
        train_time_start = datetime.datetime.now()
    # At the end of an epoch, create a savepoint checkpoint
    # which will be used to resume training in the event of preemption or
    # crashes. Intermediate checkpoints can still be used to
    # select the best checkpoint.
    epoch_checkpoint_name = model_utils.save_checkpoint(
        checkpoint, out_dir, [eval_loss] + eval_metrics,
        write_checkpoint_metrics)
    with tf.io.gfile.GFile(epoch_checkpoint, 'w') as f:
      logging.info('Epoch checkpoint: %s %s', epoch_checkpoint_name, epoch + 1)
      f.write(f'{epoch_checkpoint_name}\t{epoch}')


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
  train(_TEACHER_MODEL_DIR.value, _OUT_DIR.value, FLAGS.params, _TPU.value,
        _TPU_TOPOLOGY.value, _WRITE_CHECKPOINT_METRICS.value, _DEBUG.value)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'teacher_model_dir',
      'params',
      'out_dir',
  ])
  app.run(main)
