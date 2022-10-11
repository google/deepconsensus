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
    --ccs_bam=ccs.bam \
    --checkpoint=saved_model_directory \
    --output=predictions.fastq
"""

import concurrent.futures
import dataclasses
import enum
import itertools
import multiprocessing
import os
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_utils
from deepconsensus.postprocess import stitch_utils
from deepconsensus.preprocess import pre_lib
from deepconsensus.quality_calibration import calibration_lib
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
flags.DEFINE_string('ccs_bam', None, 'Input BAM containing ccs sequences.')

# Outputs:
flags.DEFINE_string(
    'output', None,
    'Filename of output. Use .fq or .fastq suffix to output FASTQ, '
    'or use .bam to output bam file.')

# Model checkpoint:
flags.DEFINE_string(
    'checkpoint', None, 'Path to either a checkpoint directory + prefix '
    '(e.g. "/path/to/model_directory/checkpoint-50"), '
    'or to a saved model directory, (e.g. "/path/to/model_directory") '
    'which is the directory that contains a saved_model.pb')
config_flags.DEFINE_config_file(
    'params', None, 'params.json configuration file. By default, '
    '/path/to/model_directory/params.json is used.')

# The following parameters are used at the end for filtering the final output.
flags.DEFINE_integer('min_length', 0, 'Minimum length for reads output.')
flags.DEFINE_integer('min_quality', 20, 'Minimum quality for reads output.')

# The following parameters affect performance of this script.
flags.DEFINE_integer(
    'batch_size', 1024,
    'Number of examples to batch together for TensorFlow model prediction.')
flags.DEFINE_integer(
    'batch_zmws', 100, 'Number of ZMWs to process at the same time. '
    'If 0, process all ZMWs in one batch.')
flags.DEFINE_integer(
    'skip_windows_above', 45,
    'Average CCS Base Quality used to skip individual windows from being '
    'processed by the neural network. This can help speed up DeepConsensus. '
    'Use 0 for no skipping.')
flags.DEFINE_integer(
    'ins_trim', 5, 'Trim insertions in subreads.'
    'No trimming if flag is set to 0')

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
    'If 0, then preprocessing will be done in the main process '
    'instead of using multiple processes. '
    'This flag does not control how many CPUs the model prediction '
    '(TensorFlow) uses. If you need to control that, please consider using '
    'numactl in Linux. Or if you are using Docker, considering using '
    'https://docs.docker.com/config/containers/resource_constraints/'
    '#configure-the-default-cfs-scheduler.')

# The following parameters are for TensorFlow ops device placement.
flags.DEFINE_integer(
    'use_only_gpu_index', None,
    'If set, this flag will be used for `tf.device` to specify which GPU '
    'to place the ops on. For example, if you have 3 GPU and only want to run '
    'on the 3rd GPU, you can set this to 2. By default, if you have GPUs, the '
    'lowest index one would be used.')

# The following parameters are for quality score calibration

flags.DEFINE_string(
    'dc_calibration', None, 'If set to None, base quality values will be read '
    'from model params.json if available. Set to "skip" to perform no quality '
    'calibration. Otherwise, calibration values can be directly supplied as a '
    'comma separated set of values of the linear transformation model\'s '
    'calibration values for deepconsensus base qualities. The values are set as'
    ' \"threshold,w,b\" where threshold is minimum base quality threshold '
    'after which  the linear transformation will be applied, w is the '
    'co-efficient value and b is the bias term for linear transformation. '
    'Default: None [read from params.json if available].')
flags.DEFINE_string(
    'ccs_calibration', 'skip', 'Comma separated values of '
    'linear transformation model\'s calibration values for deepconsensus base '
    'qualities. The values are set as \"threshold,w,b\" where threshold is '
    'minimum base quality threshold after which  the linear transformation '
    'will be applied, w is the co-efficient value and b is the bias term for '
    'linear transformation. Set to "skip" to perform no quality '
    'calibration. Default: "skip".')


def register_required_flags():
  flags.mark_flags_as_required([
      'subreads_to_ccs',
      'ccs_bam',
      'checkpoint',
      'output',
  ])


@dataclasses.dataclass
class InferenceOptions:
  """A central place to define options used across various stages of inference.

  Attributes:
    example_height: Height of examples, which depends on max_passes.

    max_length: Length of window.
    max_passes: Max number of subreads to include in input shown to model.
    min_quality: Quality threshold to filter final reads.
    min_length: Length threshold to filter final reads.
    batch_size: Number of examples passed through model at once.
    cpus: Number of processes to use for multiprocessing. Must be positive (for
      multiprocessing) or 0 (for serial execution).
    skip_windows_above: Run the model only when the avg(ccs_base_qual) of the
      window is below this value.
    use_saved_model: True if the given checkpoint is a saved model, false if it
      is a regular checkpoint.
    dc_calibration_values: QualityCalibrationValues defining values to be used
      for deepconsensus quality calibration.
    ccs_calibration_values: QualityCalibrationValues defining values to be used
      for ccs quality calibration.
  """
  max_length: int
  example_height: int
  max_passes: int
  min_quality: int
  min_length: int
  batch_size: int
  cpus: int
  skip_windows_above: int
  use_saved_model: bool
  dc_calibration_values: calibration_lib.QualityCalibrationValues
  ccs_calibration_values: calibration_lib.QualityCalibrationValues


timing = []


def timelog(stage: str,
            item: str,
            before: float,
            num_examples: Optional[int] = None,
            num_subreads: Optional[int] = None,
            num_zmws: Optional[int] = None) -> None:
  """Catalogue time elapsed for a given stage relative to "before"."""
  after = time.time()
  datum = {
      'item': item,
      'stage': stage,
      'runtime': after - before,
      'num_zmws': num_zmws,
      'num_examples': num_examples,
      'num_subreads': num_subreads
  }
  timing.append(datum)


# TODO Add unit test for this function. We need to create unit test
# infrastructure that allows to easily create input data for unit tests.
def batch_examples(feature_dicts: List[Tuple[str, Union[np.ndarray, int, bytes,
                                                        float]]],
                   model_params: Union[config_dict.ConfigDict,
                                       config_dict.FrozenConfigDict],
                   options: InferenceOptions):
  """Stack values for each feature.

  Args:
    feature_dicts: List of feature dictionaries.
    model_params: Parameters for the model.
    options: Some options that apply to various stages of the inference run.

  Yields:
    For Dictionary of batched features.
  """

  def process_feature_dicts(features):
    return data_providers.process_feature_dict(
        features=features, params=model_params)

  def split_list(l, batch_size):
    for i in range(0, len(l), batch_size):
      yield l[i:i + batch_size]

  processed_feature_dicts = list(map(process_feature_dicts, feature_dicts))
  for one_batch in split_list(processed_feature_dicts, options.batch_size):
    examples = {}
    for key in dc_constants.DC_FEATURES:
      vals = [x[key] for x in one_batch]
      if vals:
        examples[key] = np.stack(vals)
      else:
        examples[key] = vals
    yield examples


def run_model_on_examples(
    feature_dicts: List[Tuple[str, Union[np.ndarray, int, bytes]]],
    model: tf.keras.Model,
    model_params: Union[config_dict.ConfigDict, config_dict.FrozenConfigDict],
    options: InferenceOptions,
) -> List[stitch_utils.DCModelOutput]:
  """Runs the model on one example to get one predicted output sequence.

  Args:
    feature_dicts: List of feature dictionaries.
    model: An initialized model that will be used to make predictions.
    model_params: Parameters for the model.
    options: Some options that apply to various stages of the inference run.

  Returns:
    A DeepConsensusInput proto containing the prediction from the model.
  """

  predictions = []
  for data in batch_examples(feature_dicts, model_params, options):
    window_pos_arr = data['window_pos']
    molecule_name_arr = data['name']
    rows = tf.convert_to_tensor(data['rows'])
    if options.use_saved_model:
      softmax_output = model.signatures['serving_default'](rows)
      softmax_output = softmax_output['output_1']
    else:
      softmax_output = model.predict(rows)

    softmax_output = softmax_output.numpy()

    ec_arr = data['ec']
    np_num_passes_arr = data['np_num_passes']
    rq_arr = data['rq']
    rg_arr = data['rg']

    y_preds = np.argmax(softmax_output, -1)
    error_prob = 1 - np.max(softmax_output, axis=-1)
    quality_scores = -10 * np.log10(error_prob)
    if options.dc_calibration_values.enabled:
      quality_scores = calibration_lib.calibrate_quality_scores(
          quality_scores, options.dc_calibration_values)
    quality_scores = np.minimum(quality_scores, dc_constants.MAX_QUAL)
    quality_scores = np.round(quality_scores, decimals=0)
    quality_scores = quality_scores.astype(dtype=np.int32)
    for y_pred, qs, window_pos, molecule_name, ec, np_, rq, rg in zip(
        y_preds, quality_scores, window_pos_arr, molecule_name_arr, ec_arr,
        np_num_passes_arr, rq_arr, rg_arr):
      dc_output = stitch_utils.DCModelOutput(
          window_pos=window_pos,
          molecule_name=molecule_name,
          ec=ec,
          np_num_passes=np_,
          rq=rq,
          rg=rg)
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
      max_length=options.max_length,
      min_quality=options.min_quality,
      min_length=options.min_length,
      outcome_counter=outcome_counter)

  return fastq_string


def stream_bam(
    subreads_to_ccs: str, ccs_bam: str, options: InferenceOptions
) -> Generator[Tuple[str, str, Sequence[Any]], None, None]:
  """Streams inputs from FASTA and BAM concurrently.

  Args:
    subreads_to_ccs: Path to input BAM file with subreads aligned to template
      sequences.
    ccs_bam: Path to the input CCS BAM with template sequences (e.g.
      CCS or POA).
    options: Inference options, used to initialize a DcConfig object.

  Yields:
    For every ZMW, (ZMW name, template sequence, list of subreads).
  """

  dc_config = pre_lib.DcConfig(
      max_passes=options.max_passes, max_length=options.max_length)

  # Temporarily disable unused-variable.
  # pylint: disable=unused-variable
  proc_feeder, main_counter = pre_lib.create_proc_feeder(
      subreads_to_ccs=subreads_to_ccs,
      ccs_bam=ccs_bam,
      dc_config=dc_config,
      ins_trim=FLAGS.ins_trim)
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

  logging.info('Loading %s', checkpoint_path)
  if options.use_saved_model:
    model = tf.saved_model.load(checkpoint_path)
  else:
    model = model_utils.get_model(params)
    # This loads a model saved in tf.train.Checkpoint format through the custom
    # training loop code.
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

  model_utils.modify_params(
      params=params,
      speedy=True,
      max_length=options.max_length,
      is_training=False)
  logging.info('Finished initialize_model.')
  return model, params


def preprocess(
    one_zmw: Tuple[str, List[pre_lib.Read], pre_lib.DcConfig]
) -> List[Dict[str, Any]]:
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

  dc_whole_zmw = pre_lib.subreads_to_dc_example(
      subreads=subreads, ccs_seqname=zmw, dc_config=dc_config)
  if dc_whole_zmw is None or FLAGS.end_after_stage == DebugStage.DC_INPUT:
    return []

  # One feature dictionary per window/example.
  feature_dicts = [x.to_features_dict() for x in dc_whole_zmw.iter_examples()]
  return feature_dicts


def process_skipped_window(
    feature_dict: Dict[str, Any],
    options: InferenceOptions) -> stitch_utils.DCModelOutput:
  """Process a window by simply adopting the CCS sequence and base qualities."""
  rows = feature_dict['subreads']
  ccs = rows[-5, :, 0]
  ccs_seq = utils.encoded_sequence_to_string(ccs)
  ccs_quality_scores = feature_dict['ccs_base_quality_scores']
  if options.ccs_calibration_values.enabled:
    ccs_quality_scores = calibration_lib.calibrate_quality_scores(
        ccs_quality_scores, options.ccs_calibration_values)
  ccs_quality_scores = np.minimum(ccs_quality_scores, dc_constants.MAX_QUAL)
  ccs_quality_scores = ccs_quality_scores.astype(dtype=np.int32)
  dc_output = stitch_utils.DCModelOutput(
      window_pos=feature_dict['window_pos'],
      molecule_name=feature_dict['name'],
      sequence=ccs_seq,
      quality_string=utils.quality_scores_to_string(ccs_quality_scores),
      ec=feature_dict['ec'],
      np_num_passes=feature_dict['np_num_passes'],
      rq=feature_dict['rq'],
      rg=feature_dict['rg'])
  return dc_output


def inference_on_n_zmws(
    inputs: Sequence[Tuple[str, str, Sequence[Any]]],
    model: tf.keras.Model,
    model_params: Union[config_dict.ConfigDict, config_dict.FrozenConfigDict],
    output_writer: Union[gfile.GFile, pysam.AlignmentFile],
    options: InferenceOptions,
    batch_name: str,
    outcome_counter: stitch_utils.OutcomeCounter,
    pool: Optional[concurrent.futures.ProcessPoolExecutor] = None) -> None:
  """Runs the full inference process on a batch of ZMWs and writes to fastq.

  Args:
    inputs: Iterable of inputs, one for each ZMW, each of which has
        three elements: (name of zmw, aligned_subreads, DcConfig).
    model: An initialized model that will be used to make predictions.
    model_params: Parameters for the model.
    output_writer: File writer where fastq or bam output will be written.
    options: Some options that apply to various stages of the inference run.
    batch_name: Name of batch used for runtime metrics.
    outcome_counter: Counts outcomes for each ZMW.
    pool: Process pool to run the preprocessing on. If None or empty,
        preprocessing will be done sequentially on the main process.
  """
  before_batch = time.time()

  if options.cpus == 0:
    # Preprocess ZMWs one at a time in the main process without multiprocessing.
    outputs = [preprocess(one_zmw=one_zmw) for one_zmw in inputs]
  else:
    assert pool
    # Each call to preprocess gets one ZMW from inputs.
    outputs = list(pool.map(preprocess, inputs))

  feature_dicts_for_zmws = outputs
  num_zmws = len(feature_dicts_for_zmws)

  batch_total_examples = sum([len(zmw) for zmw in feature_dicts_for_zmws])
  batch_total_subreads = sum([len(subreads) for _, subreads, _ in inputs])
  timelog(
      stage='preprocess',
      item=batch_name,
      before=before_batch,
      num_examples=batch_total_examples,
      num_subreads=batch_total_subreads,
      num_zmws=num_zmws)
  if FLAGS.end_after_stage in [DebugStage.TF_EXAMPLES, DebugStage.DC_INPUT]:
    return

  before = time.time()
  before_skipping = time.time()

  if options.skip_windows_above:
    # Skip windows with average CCS predicted qualities above threshold.
    feature_dicts_for_model = []
    predictions_for_skipped_windows = []
    for one_zmw in feature_dicts_for_zmws:
      for window in one_zmw:
        avg_ccs_base_quality = utils.avg_phred(
            window['ccs_base_quality_scores'])
        if avg_ccs_base_quality <= options.skip_windows_above:
          feature_dicts_for_model.append(window)
        else:
          dc_output_for_window = process_skipped_window(window, options)
          predictions_for_skipped_windows.append(dc_output_for_window)
  else:
    # Go straight to model without skipping any windows.
    feature_dicts_for_model = []
    for one_zmw in feature_dicts_for_zmws:
      for window in one_zmw:
        feature_dicts_for_model.append(window)
    predictions_for_skipped_windows = []

  time_to_skip = time.time() - before_skipping

  before_run_model = time.time()
  predictions_from_model = run_model_on_examples(feature_dicts_for_model, model,
                                                 model_params, options)
  time_to_run_model = time.time() - before_run_model

  predictions = predictions_from_model + predictions_for_skipped_windows

  def percent_of_examples(numerator):
    if not predictions:
      return 0  # Avoid dividing by zero.
    return 100 * (numerator / len(predictions))

  logging.info(
      'Example summary: ran model=%d (%0.2f%%; %0.3fs) skip=%d (%0.2f%%; %0.3fs) total=%d.',
      len(predictions_from_model),
      percent_of_examples(len(predictions_from_model)), time_to_run_model,
      len(predictions_for_skipped_windows),
      percent_of_examples(len(predictions_for_skipped_windows)), time_to_skip,
      len(predictions))

  timelog(
      stage='run_model',
      item=batch_name,
      before=before,
      num_examples=batch_total_examples,
      num_subreads=batch_total_subreads,
      num_zmws=num_zmws)
  if FLAGS.end_after_stage == DebugStage.RUN_MODEL:
    return

  before = time.time()
  # Sort predictions prior to grouping
  # pylint: disable=g-long-lambda
  predictions = sorted(
      predictions, key=lambda dc: (dc.molecule_name, dc.window_pos))

  for zmw, predictions_for_zmw in itertools.groupby(predictions,
                                                    lambda p: p.molecule_name):
    predictions_for_zmw = list(predictions_for_zmw)
    fastq_string = stitch_utils.stitch_to_fastq(
        molecule_name=zmw,
        predictions=predictions_for_zmw,
        max_length=options.max_length,
        min_quality=options.min_quality,
        min_length=options.min_length,
        outcome_counter=outcome_counter)

    if fastq_string:
      # FASTQs are written with gfile, bams are written with pysam.
      if isinstance(output_writer, gfile.GFile):
        output_writer.write(fastq_string)
      else:
        name, seq, _, qual = fastq_string.splitlines()
        # Remove the @ prefix from sequence name.
        name = name[1:]
        record = pysam.AlignedSegment()
        record.query_name = name
        record.query_sequence = seq
        record.query_qualities = pysam.qualitystring_to_array(qual)
        record.flag = 4  # unmapped.
        record.mapping_quality = 255
        zmw = int(name.split('/')[1])
        record.set_tags([
            ('ec', predictions_for_zmw[0].ec or -1, 'f'),
            ('np', predictions_for_zmw[0].np_num_passes, 'i'),
            ('rq', predictions_for_zmw[0].rq, 'f'),
            ('RG', predictions_for_zmw[0].rg, 'Z'),
            ('zm', zmw, 'i'),
        ])
        output_writer.write(record)

  timelog(
      stage='stitch_and_write_fastq',
      item=batch_name,
      before=before,
      num_examples=batch_total_examples,
      num_subreads=batch_total_subreads,
      num_zmws=num_zmws)
  logging.info('Processed a batch of %d ZMWs in %0.3f seconds', len(inputs),
               time.time() - before_batch)


def save_runtime(time_points, output_prefix):
  """Save CSV of runtime."""
  df = pd.DataFrame(time_points)

  # Save CSV to file.
  with tf.io.gfile.GFile(f'{output_prefix}.csv', 'w') as writer:
    df.to_csv(writer, index=False)


def run() -> stitch_utils.OutcomeCounter:
  """Performs an inference run."""

  # Determine if --checkpoint is a saved model.
  use_saved_model = (
      tf.io.gfile.exists(FLAGS.checkpoint) and
      tf.io.gfile.exists(f'{FLAGS.checkpoint}/saved_model.pb'))

  # Load model parameters
  if not FLAGS.params:
    params = model_utils.read_params_from_json(checkpoint_path=FLAGS.checkpoint)
  else:
    params = FLAGS.params

  dc_config = pre_lib.DcConfig(params.max_passes, params.max_length)

  # Attempt to read default calibration values from model params.json.
  # If not found, set to 'skip'.
  if FLAGS.dc_calibration is None:
    dc_calibration_values = params.get('dc_calibration', 'skip')
    if dc_calibration_values != 'skip':
      logging.info(
          'DeepConsensus base calibration values read from '
          'model params.json: %s', dc_calibration_values)
  else:
    dc_calibration_values = FLAGS.dc_calibration
  dc_calibration_values = calibration_lib.parse_calibration_string(
      dc_calibration_values)
  if not FLAGS.ccs_calibration:
    raise ValueError('--ccs_calibration should be set to "skip" '
                     'or to base calibration scores.')
  ccs_calibration_values = calibration_lib.parse_calibration_string(
      FLAGS.ccs_calibration)

  options = InferenceOptions(
      max_length=params.max_length,
      example_height=dc_config.tensor_height,
      max_passes=params.max_passes,
      min_quality=FLAGS.min_quality,
      min_length=FLAGS.min_length,
      batch_size=FLAGS.batch_size,
      cpus=FLAGS.cpus,
      skip_windows_above=FLAGS.skip_windows_above,
      use_saved_model=use_saved_model,
      dc_calibration_values=dc_calibration_values,
      ccs_calibration_values=ccs_calibration_values)
  outcome_counter = stitch_utils.OutcomeCounter()

  pool = None
  if options.cpus > 0:
    # Spin up multiple processes, each taking the next ZMW when ready.
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=options.cpus)
    logging.info('Using multiprocessing: cpus is %s.', options.cpus)
  elif options.cpus < 0:
    raise ValueError('Number of processes must be positive '
                     '(for multiprocessing) or 0 (for serial execution).')

  # Set up model.
  before_model_setup = time.time()
  loaded_model, model_params = initialize_model(
      checkpoint_path=FLAGS.checkpoint, params=params, options=options)
  logging.info('Model setup took %s seconds.', time.time() - before_model_setup)

  # Initialize output fastq writer.
  output_fname = FLAGS.output
  correct_suffix = output_fname.endswith('.fq') or output_fname.endswith(
      '.fastq') or output_fname.endswith('.bam')
  if not correct_suffix:
    raise NameError('Filename must end in .fq, .fastq, or .bam')

  output_dir = os.path.dirname(output_fname)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)

  if output_fname.endswith('.fq') or output_fname.endswith('.fastq'):
    output_writer = gfile.Open(output_fname, 'wb')
  else:
    ccs_bam_header = pysam.AlignmentFile(FLAGS.ccs_bam, check_sq=False).header
    output_writer = pysam.AlignmentFile(
        output_fname, 'wb', header=ccs_bam_header)

  input_file_generator = stream_bam(
      subreads_to_ccs=FLAGS.subreads_to_ccs,
      ccs_bam=FLAGS.ccs_bam,
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
          output_writer=output_writer,
          options=options,
          batch_name=str(batch_count),
          outcome_counter=outcome_counter,
          pool=pool)
      batch_count += 1
      stored_n_zmws = []
      logging.info('Processed %s ZMWs in %0.3f seconds', zmw_counter,
                   time.time() - before_all_zmws)

  if stored_n_zmws:
    inference_on_n_zmws(
        inputs=stored_n_zmws,
        model=loaded_model,
        model_params=model_params,
        output_writer=output_writer,
        options=options,
        batch_name=str(batch_count),
        outcome_counter=outcome_counter,
        pool=pool)

  if pool:
    pool.shutdown(wait=True)

  output_writer.close()

  logging.info('Processed %s ZMWs in %0.3f seconds', zmw_counter,
               time.time() - before_all_zmws)
  logging.info('Outcome counts: %s', outcome_counter)
  save_runtime(time_points=timing, output_prefix=f'{output_fname}.runtime')
  return outcome_counter


def main(_):
  """Main entry point."""
  if FLAGS.ccs_fasta:
    raise NotImplementedError('The --ccs_fasta flag has been deprecated. '
                              'Please use --ccs_bam instead.')
  if FLAGS.use_only_gpu_index:
    with tf.device(f'GPU:{FLAGS.use_only_gpu_index}'):
      outcome_counter = run()
  else:
    outcome_counter = run()
  if not outcome_counter.success:
    return 1  # indicating an error has occurred.

if __name__ == '__main__':
  register_required_flags()
  app.run(main)
