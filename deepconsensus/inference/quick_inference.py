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
r"""Run DeepConsensus and generate a polished FASTQ.

Usage:
  deepconsensus run \
    --subreads_to_ccs=subreads_to_ccs.bam \
    --ccs_fasta=ccs_fasta.fasta \
    --output=predictions.fastq \
    --cpus=4

"""
import dataclasses
import enum
import itertools
import multiprocessing
import os
import time
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
import numpy as np
import pandas as pd
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_utils
from deepconsensus.postprocess import stitch_utils
from deepconsensus.preprocess import utils as preprocess_utils
from deepconsensus.utils import dc_constants
from deepconsensus.utils import utils
from tensorflow.python.platform import gfile


@enum.unique
class DebugStage(enum.Enum):
  """Stage to end after for debugging and runtime testing purposes."""
  DC_INPUT = 1
  TF_EXAMPLES = 2
  RUN_MODEL = 3
  FULL = 4


FLAGS = flags.FLAGS

# Inputs:
flags.DEFINE_string('subreads_to_ccs', None,
                    'Input BAM containing subreads aligned to ccs.')
flags.DEFINE_string('ccs_fasta', None, 'Input FASTA containing ccs sequences.')

# Outputs:
flags.DEFINE_string(
    'output', None, 'Filename of output FASTQ file. If this path '
    'does not end in ".fastq", the suffix will be added.')

# Model checkpoint:
flags.DEFINE_string(
    'checkpoint', None, 'Path to checkpoint directory + prefix. '
    'For example: path/to/model/checkpoint-50.')
config_flags.DEFINE_config_file('params', None,
                                'params.json configuration file.')

# The following just need to match the training parameters.
flags.DEFINE_integer('max_passes', 20, 'Maximum subreads in each input.')
flags.DEFINE_integer('example_width', 100, 'Number of bases in each input.')
flags.DEFINE_integer(
    'padding', 20, 'Number of bases of padding to add to example_width to '
    'allow for insertions.')

# The following parameters are used at the end for filtering the final output.
flags.DEFINE_integer('min_length', 0, 'Minimum length for reads output.')
flags.DEFINE_integer('min_quality', 20, 'Minimum quality for reads output.')

# The following parameters affect performance of this script.
flags.DEFINE_integer(
    'batch_size', 1024,
    'Number of examples to batch together for TensorFlow model prediction.')
flags.DEFINE_integer(
    'batch_zmws', 20, 'Number of ZMWs to process at the same time. '
    'If 0, process all ZMWs in one batch.')

# The following parameters are for debugging.
flags.DEFINE_integer('limit', None, 'Only process this many ZMWs. ')

flags.DEFINE_enum_class(
    'end_after_stage', 'full', DebugStage,
    'For debugging and runtime measurement purposes, '
    'end after this stage for each ZMW.')
flags.DEFINE_integer(
    'cpus',
    multiprocessing.cpu_count() - 1,
    'Number of processes to use during preprocessing stage. '
    'Uses CPU count - 1 by default. '
    ' If 0, then preprocessing will be done in the main process '
    'instead of using multiple processes.')


def register_required_flags():
  flags.mark_flags_as_required([
      'subreads_to_ccs',
      'ccs_fasta',
      'checkpoint',
      'output',
  ])


@dataclasses.dataclass
class InferenceOptions:
  """A central place to define options used across various stages of inference.

  Attributes:
    example_width: Number of bases for each window/example given to the model.
    example_height: Height of examples, which depends on max_passes.
    padding: Number of bases of padding to add to example_width to allow for
      insertions.
    padded_len: Length of window after padding is added. This should be equal to
      example_width + padding.
    max_passes: Max number of subreads to include in input shown to model.
    min_quality: Quality threshold to filter final reads.
    min_length: Length threshold to filter final reads.
    batch_size: Number of examples passed through model at once.
    cpus: Number of processes to use for multiprocessing. Must be
      positive (for multiprocessing) or 0 (for serial execution).
  """
  example_width: int
  example_height: int
  padding: int
  padded_len: int
  max_passes: int
  min_quality: int
  min_length: int
  batch_size: int
  cpus: int


timing = []


def timelog(stage: str,
            item: str,
            before: float,
            num_examples: Optional[int] = None,
            num_subreads: Optional[int] = None,
            is_batch: bool = True,
            update_global_variable: bool = True) -> Dict[str, Any]:
  """Catalogue time elapsed for a given stage relative to "before"."""
  after = time.time()
  datum = {
      'item': item,
      'stage': stage,
      'start_time': before,
      'end_time': after,
      'runtime': after - before,
      'is_batch': is_batch
  }
  if num_examples:
    datum['num_examples'] = num_examples
  if num_subreads:
    datum['num_subreads'] = num_subreads
  if update_global_variable:
    timing.append(datum)
  return datum


def run_model_on_examples(
    feature_dict_gen_fn: Callable[[], Dict[str, Union[np.ndarray, int, bytes]]],
    model: tf.keras.Model,
    model_params: Union[config_dict.ConfigDict, config_dict.FrozenConfigDict],
    options: InferenceOptions,
) -> List[stitch_utils.DCModelOutput]:
  """Runs the model on one example to get one predicted output sequence.

  Args:
    feature_dict_gen_fn: Generator fn of feature dictionaries.
    model: An initialized model that will be used to make predictions.
    model_params: Parameters for the model.
    options: Some options that apply to various stages of the inference run.

  Returns:
    A DeepConsensusInput proto containing the prediction from the model.
  """
  def _process_input_helper(
      features: Dict[str, tf.Tensor]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return data_providers.process_feature_dict(
        features=features, params=model_params)

  dataset = tf.data.Dataset.from_generator(
      feature_dict_gen_fn,
      output_signature={
          'subreads':
              tf.TensorSpec(
                  shape=(options.example_height, model_params.max_length,
                         model_params.num_channels),
                  dtype=dc_constants.TF_DATA_TYPE),
          'subreads/num_passes':
              tf.TensorSpec(shape=(), dtype=tf.int32),
          'name':
              tf.TensorSpec(shape=(), dtype=tf.string),
          'window_pos':
              tf.TensorSpec(shape=(), dtype=tf.int32),
      })
  dataset = dataset.map(map_func=_process_input_helper)
  dataset = dataset.batch(batch_size=options.batch_size, drop_remainder=False)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  predictions = []
  for rows, _, _, window_pos_arr, molecule_name_arr in dataset.as_numpy_iterator(
  ):
    softmax_output = model.predict(rows)
    y_preds = tf.argmax(softmax_output, -1)
    error_prob = 1 - np.max(softmax_output, axis=-1)
    quality_scores = -10 * np.log10(error_prob)
    # Round to the nearest integer and cap at max allowed value.
    quality_scores = np.round(quality_scores, decimals=0)
    quality_scores = np.minimum(quality_scores, dc_constants.MAX_QUAL)
    quality_scores = quality_scores.astype(dtype=np.int32)
    for y_pred, qs, window_pos, molecule_name in zip(y_preds, quality_scores,
                                                     window_pos_arr,
                                                     molecule_name_arr):
      dc_output = stitch_utils.DCModelOutput(
          window_pos=window_pos, molecule_name=molecule_name.decode('utf=8'))
      y_pred_bases = ''.join(
          np.vectorize(dc_constants.VOCAB.__getitem__)(y_pred))
      quality_string = utils.quality_scores_to_string(qs)
      dc_output.sequence = y_pred_bases
      dc_output.quality_string = quality_string
      predictions.append(dc_output)
  return predictions


def stitch_predictions_for_one_zmw(
    predictions: Iterable[stitch_utils.DCModelOutput],
    zmw: str,
    options: InferenceOptions,
    outcome_counter=stitch_utils.OutcomeCounter) -> Optional[str]:
  """Stitches together predictions into one sequence.

  Args:
    predictions: Predictions from running model on examples.
    zmw: Molecule name, the part that is shared among all subreads.
    options: Options here are used for filtering.
    outcome_counter: Keeps track of how many ZMWs end up with which outcomes.

  Returns:
    Fastq string for one sequence.
  """
  fastq_string = stitch_utils.stitch_to_fastq(
      molecule_name=zmw,
      predictions=predictions,
      example_width=options.example_width,
      min_quality=options.min_quality,
      min_length=options.min_length,
      outcome_counter=outcome_counter)

  return fastq_string


def stream_fasta_and_bam(
    subreads_to_ccs: str, ccs_fasta: str, options: InferenceOptions
) -> Generator[Tuple[str, str, Sequence[Any]], None, None]:
  """Streams inputs from FASTA and BAM concurrently.

  Args:
    subreads_to_ccs: Path to input BAM file with subreads aligned to template
      sequences.
    ccs_fasta: Path to the input FASTA file with template sequences (e.g.
      CCS or POA).
    options: Inference options, used to initialize a DcConfig object.

  Yields:
    For every ZMW, (ZMW name, template sequence, list of subreads).
  """

  dc_config = preprocess_utils.DcConfig(
      max_passes=options.max_passes,
      example_width=options.example_width,
      padding=options.padding)

  # Temporarily disable unused-variable.
  # pylint: disable=unused-variable
  proc_feeder, main_counter = preprocess_utils.create_proc_feeder(
      subreads_to_ccs=subreads_to_ccs, ccs_fasta=ccs_fasta, dc_config=dc_config)
  # pylint: enable=unused_variable

  for input_data in proc_feeder():
    subreads, zmw, dc_config, _ = input_data
    yield zmw, subreads, dc_config


def initialize_model(
    checkpoint_path: str, params: config_dict.ConfigDict,
    options: InferenceOptions
) -> Tuple[Optional[tf.keras.Model], Optional[config_dict.ConfigDict]]:
  """Initializes the model and gathers parameters.

  Args:
    checkpoint_path: Path to model checkpoint.
    params: Parameter object, from flags.
    options: Contains a few more parameters some of which will replace those in
      the params object.

  Returns:
    A tuple containing an initialized model and a final parameter set.
  """
  if FLAGS.end_after_stage in [DebugStage.TF_EXAMPLES, DebugStage.DC_INPUT]:
    return None, None

  # Figure out model parameters.
  if not FLAGS.params:
    params = model_utils.read_params_from_json(checkpoint_path=checkpoint_path)
  else:
    params = FLAGS.params

  with params.unlocked():
    params.max_passes = options.max_passes
  logging.info('Loading %s', checkpoint_path)
  model = model_utils.get_model(params)
  # This loads a model saved in tf.train.Checkpoint format through the custom
  # training loop code.
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_path).expect_partial()

  model_utils.modify_params(
      params=params,
      speedy=True,
      max_length=options.padded_len,
      is_training=False)
  logging.info('Finished initialize_model.')
  return model, params


def preprocess(
    one_zmw: Tuple[str, List[preprocess_utils.Read], preprocess_utils.DcConfig]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
  """Preprocess input data for one ZMW into windows of features.

  This is often run from multiple processes in parallel, which creates some
  constraints to keep in mind. Adjustments include returning runtime data
  points instead of updating a global variable.

  Args:
    one_zmw: Input data for one ZMW: a tuple of (name, subreads, DcConfig).

  Returns:
    A list of feature dictionaries, one for each window.
    A list of time log dictionaries, one for each of the two stages.
  """
  zmw, subreads, dc_config = one_zmw

  time_logs = []
  before = time.time()
  dc_whole_zmw = preprocess_utils.subreads_to_dc_example(
      subreads=subreads, ccs_seqname=zmw, dc_config=dc_config)
  time_logs.append(
      timelog(
          stage='make_dc_input',
          item=zmw,
          before=before,
          num_subreads=len(subreads),
          is_batch=False,
          update_global_variable=False))
  if dc_whole_zmw is None or FLAGS.end_after_stage == DebugStage.DC_INPUT:
    return [], time_logs

  before = time.time()

  # One feature dictionary per window/example.
  feature_dicts = [x.to_features_dict() for x in dc_whole_zmw.iter_examples()]
  time_logs.append(
      timelog(
          stage='make_tf_examples',
          item=zmw,
          before=before,
          num_examples=len(feature_dicts),
          is_batch=False,
          update_global_variable=False))
  return feature_dicts, time_logs


def inference_on_n_zmws(inputs: Sequence[Tuple[str, str, Sequence[Any]]],
                        model: tf.keras.Model,
                        model_params: Union[config_dict.ConfigDict,
                                            config_dict.FrozenConfigDict],
                        fastq_writer: gfile.GFile,
                        options: InferenceOptions,
                        batch_name: str,
                        outcome_counter=stitch_utils.OutcomeCounter) -> None:
  """Runs the full inference process on a batch of ZMWs and writes to fastq.

  Args:
    inputs: Iterable of inputs, one for each ZMW, each of which has
        three elements: (name of zmw, aligned_subreads, DcConfig).
    model: An initialized model that will be used to make predictions.
    model_params: Parameters for the model.
    fastq_writer: File writer where fastq output will be written.
    options: Some options that apply to various stages of the inference run.
    batch_name: Name of batch used for runtime metrics.
    outcome_counter: Counts outcomes for each ZMW.
  """
  before_batch = time.time()

  if options.cpus == 0:
    # Preprocess ZMWs one at a time in the main process without multiprocessing.
    outputs = [preprocess(one_zmw=one_zmw) for one_zmw in inputs]
  elif options.cpus > 0:
    logging.log_first_n(logging.INFO,
                        f'Using multiprocessing: cpus is {options.cpus}.', 1)
    # Spin up multiple processes, each taking the next ZMW when ready.
    pool = multiprocessing.Pool(processes=options.cpus)

    # Each call to preprocess gets one ZMW from inputs.
    outputs = pool.map(preprocess, inputs)
    pool.close()
    logging.vlog(
        1, 'Multiprocessing pool is done and closed. '
        'Number of outputs: %d', len(outputs))
  else:
    raise ValueError('Number of processes must be positive '
                     '(for multiprocessing) or 0 (for serial execution).')

  # Unpack outputs into two lists.
  feature_dicts_for_zmws, time_logs = zip(*outputs)

  # Unpack list of lists of time data (e.g. 10 ZMWs, 2 time points for each).
  time_logs = list(itertools.chain(*time_logs))
  timing.extend(time_logs)

  batch_total_examples = sum([len(zmw) for zmw in feature_dicts_for_zmws])
  batch_total_subreads = sum([len(subreads) for _, subreads, _ in inputs])
  timelog(
      stage='preprocess',
      item=batch_name,
      before=before_batch,
      num_examples=batch_total_examples,
      num_subreads=batch_total_subreads)
  if FLAGS.end_after_stage in [DebugStage.TF_EXAMPLES, DebugStage.DC_INPUT]:
    return

  before = time.time()
  # Make a generator function from the list.
  def feature_dict_gen_fn():
    for feature_dicts_for_one_zmw in feature_dicts_for_zmws:
      yield from feature_dicts_for_one_zmw

  predictions = run_model_on_examples(feature_dict_gen_fn, model, model_params,
                                      options)
  timelog(stage='run_model', item=batch_name, before=before)
  if FLAGS.end_after_stage == DebugStage.RUN_MODEL:
    return

  before = time.time()
  for zmw, predictions_for_zmw in itertools.groupby(predictions,
                                                    lambda p: p.molecule_name):
    fastq_string = stitch_predictions_for_one_zmw(
        predictions=predictions_for_zmw,
        zmw=zmw,
        options=options,
        outcome_counter=outcome_counter)
    if fastq_string:
      fastq_writer.write(fastq_string)
  timelog(stage='stitch_and_write_fastq', item=batch_name, before=before)
  logging.info('Processed a batch of %d ZMWs in %s seconds', len(inputs),
               time.time() - before_batch)


def save_runtime(time_points, output_prefix):
  """Save CSV of runtime."""
  df = pd.DataFrame(time_points)

  # Show program start time as 0.
  min_time = min(df['start_time'])
  df['start_time'] -= min_time
  df['end_time'] -= min_time

  # Save CSV to file.
  with tf.io.gfile.GFile(f'{output_prefix}.csv', 'w') as writer:
    df.to_csv(writer)


def run() -> stitch_utils.OutcomeCounter:
  """Called by main."""
  dc_config = preprocess_utils.DcConfig(FLAGS.max_passes, FLAGS.example_width,
                                        FLAGS.padding)
  options = InferenceOptions(
      example_width=FLAGS.example_width,
      example_height=dc_config.tensor_height,
      padding=FLAGS.padding,
      padded_len=FLAGS.example_width + FLAGS.padding,
      max_passes=FLAGS.max_passes,
      min_quality=FLAGS.min_quality,
      min_length=FLAGS.min_length,
      batch_size=FLAGS.batch_size,
      cpus=FLAGS.cpus)
  outcome_counter = stitch_utils.OutcomeCounter()

  # Set up model.
  before_model_setup = time.time()
  loaded_model, model_params = initialize_model(
      checkpoint_path=FLAGS.checkpoint, params=FLAGS.params, options=options)
  logging.info('Model setup took %s seconds.', time.time() - before_model_setup)

  # Initialize output fastq writer.
  output_filename = FLAGS.output
  if not output_filename.endswith('.fastq'):
    output_filename += '.fastq'

  output_dir = os.path.dirname(output_filename)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  fastq_writer = gfile.Open(output_filename, 'wb')

  input_file_generator = stream_fasta_and_bam(
      subreads_to_ccs=FLAGS.subreads_to_ccs,
      ccs_fasta=FLAGS.ccs_fasta,
      options=options)

  num_zmws_to_batch = FLAGS.batch_zmws

  # Process ZMWs.
  before_all_zmws = time.time()
  zmw_counter = 0
  batch_count = 0
  stored_n_zmws = []
  for zmw, subreads, dc_config in input_file_generator:
    if FLAGS.limit and zmw_counter >= FLAGS.limit:
      break
    zmw_counter += 1
    stored_n_zmws.append((zmw, subreads, dc_config))
    if num_zmws_to_batch and len(stored_n_zmws) >= num_zmws_to_batch:
      inference_on_n_zmws(
          inputs=stored_n_zmws,
          model=loaded_model,
          model_params=model_params,
          fastq_writer=fastq_writer,
          options=options,
          batch_name=f'batch {batch_count}: {len(stored_n_zmws)} ZMWs',
          outcome_counter=outcome_counter)
      batch_count += 1
      stored_n_zmws = []
      logging.info('Processed %s ZMWs in %f seconds', zmw_counter,
                   time.time() - before_all_zmws)

  if stored_n_zmws:
    inference_on_n_zmws(
        inputs=stored_n_zmws,
        model=loaded_model,
        model_params=model_params,
        fastq_writer=fastq_writer,
        options=options,
        batch_name=f'batch {batch_count}: {len(stored_n_zmws)} ZMWs',
        outcome_counter=outcome_counter)

  fastq_writer.close()

  logging.info('Processed %s ZMWs in %s seconds', zmw_counter,
               time.time() - before_all_zmws)
  logging.info('Outcome counts: %s', outcome_counter)
  save_runtime(time_points=timing, output_prefix=f'{output_filename}.runtime')
  return outcome_counter


def main(_):
  """Main entry point."""
  outcome_counter = run()
  if not outcome_counter.success:
    return 1  # indicating an error has occurred.

if __name__ == '__main__':
  register_required_flags()
  app.run(main)
