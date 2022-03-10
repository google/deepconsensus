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
r"""Performs preprocessing of subread-aligned data.

Usage:
  deepconsensus preprocess \
    --subreads_to_ccs=subreads_to_ccs.bam \
    --ccs_fasta=ccs_fasta.fasta \
    --truth_bed=truth_bed.bed \
    --truth_to_ccs=truth_to_ccs.bam \
    --truth_split=truth_split.tsv \
    --output=examples-@split.tfrecord.gz \
    --cpus=4

"""

import collections
import functools
import json
import multiprocessing
import multiprocessing.pool
import os
import time
from typing import Dict, List

from absl import flags
from absl import logging
import tensorflow as tf

from deepconsensus.preprocess import utils
from deepconsensus.utils import dc_constants
from absl import app


AsyncResult = multiprocessing.pool.AsyncResult
Queue = multiprocessing.Queue
Issue = dc_constants.Issue


FLAGS = flags.FLAGS

flags.DEFINE_string('subreads_to_ccs', None,
                    'Input BAM containing subreads aligned to ccs.')
flags.DEFINE_string('ccs_fasta', None, 'Input FASTA containing ccs sequences.')
flags.DEFINE_string(
    'output', None,
    ('Output filename. If training, use @split wildcard for split name. '
     'For example: --output=examples-@split.tfrecord.gz'
     'The output filename must end in .tfrecord.gz'))
flags.DEFINE_string('truth_to_ccs', None, 'Input truth bam aligned to ccs.')
flags.DEFINE_string('truth_bed', None, 'Input truth bedfile.')
# TODO TODO
flags.DEFINE_string('truth_split', None,
                    'Input file defining train/eval/test splits.')
flags.DEFINE_integer(
    'cpus',
    multiprocessing.cpu_count(),
    'Number of worker processes to use. Use 0 to disable parallel processing. '
    'Minimum of 2 CPUs required for parallel processing.',
    short_name='j')
flags.DEFINE_integer('bam_reader_threads', 8,
                     'Number of decompression threads to use.')
flags.DEFINE_integer('limit', 0, 'Limit processing to n ZMWs.')


def register_required_flags():
  flags.mark_flags_as_required([
      'subreads_to_ccs',
      'ccs_fasta',
      'output',
  ])


def trace_exception(f):
  """Decorator to catch errors run in multiprocessing processes."""

  @functools.wraps(f)
  def wrap(*args, **kwargs):
    try:
      result = f(*args, **kwargs)
      return result
    except:  # pylint: disable=bare-except
      logging.exception('Error in function %s.', f.__name__)
      raise Exception('Error in worker process')

  return wrap


def make_dirs(path):
  # Create directories for filename
  tf.io.gfile.makedirs(os.path.dirname(path))


def setup_writers(output_fname: str,
                  splits: List[str]) -> Dict[str, tf.io.TFRecordWriter]:
  """Creates tf writers for split set."""
  tf_writers = {}
  tf_ops = tf.io.TFRecordOptions(compression_type='GZIP')
  for split in splits:
    split_fname = output_fname.replace('@split', split)
    # Create subdirs if necessary.
    make_dirs(split_fname)
    tf_writers[split] = tf.io.TFRecordWriter(split_fname, tf_ops)
  return tf_writers


def write_tf_record(tf_example_str_set: List[bytes], split: str,
                    tf_writers: Dict[str, tf.io.TFRecordWriter]):
  """Writes tf examples to a split."""
  for tf_example_str in tf_example_str_set:
    tf_writers[split].write(tf_example_str)
  tf_writers[split].flush()


@trace_exception
def tf_record_writer(output_fname: str, splits: List[str],
                     queue: Queue) -> bool:
  """tf_record writing worker."""
  tf_writers = setup_writers(output_fname, splits)
  while True:
    tf_example_str_set, split = queue.get()
    if split == 'kill':
      break
    write_tf_record(tf_example_str_set, split, tf_writers)
  for writer in tf_writers.values():
    writer.close()
  return True


@trace_exception
def process_subreads(subreads: List[utils.Read],
                     ccs_seqname: str,
                     dc_config: utils.DcConfig,
                     split: str,
                     queue: Queue,
                     local=False):
  """Subread processing worker."""
  tf_out = []
  dc_example = utils.subreads_to_dc_example(subreads, ccs_seqname, dc_config)
  for example in dc_example.iter_examples():
    tf_out.append(example.tf_example().SerializeToString())
  dc_example.counter[f'n_examples_{split}'] += len(tf_out)
  dc_example.counter['n_examples'] += len(tf_out)
  if local:
    return tf_out, split, dc_example.counter
  else:
    queue.put([tf_out, split])
  # Return a counter object for each ZMW.
  return dc_example.counter


def clear_tasks(tasks: List[AsyncResult],
                main_counter: collections.Counter) -> List[AsyncResult]:
  """Clear successful tasks and log result."""
  for task in tasks:
    if task.ready():
      if task.successful():
        # Fetch task results and integrate into main counter
        counter = task.get()[0]
        main_counter.update(counter)
        tasks.remove(task)
      else:
        raise Exception('A worker process failed.')
  logging.info('Processed %s ZMWs.', main_counter['n_zmw_pass'])
  return tasks


def main(unused_argv) -> None:
  if FLAGS.cpus == 1:
    raise ValueError('Must set cpus to 0 or >=2 for parallel processing.')

  is_training = FLAGS.truth_to_ccs and FLAGS.truth_bed and FLAGS.truth_split

  if not FLAGS.output.endswith('.tfrecord.gz'):
    raise ValueError('--output must end with .tfrecord.gz')

  if is_training:
    logging.info('Generating tf.Examples in training mode.')
    contig_split = utils.read_truth_split(FLAGS.truth_split)
    splits = set(contig_split.values())
    for split in splits:
      if '@split' not in FLAGS.output:
        raise ValueError('You must add @split to --output when training.')
  elif FLAGS.truth_to_ccs or FLAGS.truth_bed or FLAGS.truth_split:
    err_msg = ('You must specify truth_to_ccs, truth_bed, and truth_split '
               'to generate a training dataset.')
    raise Exception(err_msg)
  else:
    logging.info('Generating tf.Examples in inference mode.')
    splits = ['inference']

  manager = multiprocessing.Manager()
  queue = manager.Queue()

  dc_config = utils.DcConfig(max_passes=20, example_width=100, padding=20)

  proc_feeder, main_counter = utils.create_proc_feeder(
      subreads_to_ccs=FLAGS.subreads_to_ccs,
      ccs_fasta=FLAGS.ccs_fasta,
      dc_config=dc_config,
      truth_bed=FLAGS.truth_bed,
      truth_to_ccs=FLAGS.truth_to_ccs,
      truth_split=FLAGS.truth_split,
      limit=FLAGS.limit,
      bam_reader_threads=FLAGS.bam_reader_threads)

  if FLAGS.cpus == 0:
    logging.info('Using a single cpu.')
    tf_writers = setup_writers(FLAGS.output, splits)
    for args in proc_feeder():
      tf_example_str_set, split, counter = process_subreads(
          *args, queue=None, local=True)
      write_tf_record(tf_example_str_set, split, tf_writers)
      # Update counter
      main_counter.update(counter)
      if main_counter['n_zmw_pass'] % 20 == 0:
        logging.info('Processed %s ZMWs.', main_counter['n_zmw_pass'])
  else:
    logging.info('Processing in parallel using %s cores', FLAGS.cpus)
    with multiprocessing.Pool(FLAGS.cpus) as pool:
      # Setup parallelization
      tf_writer = pool.apply_async(tf_record_writer,
                                   (FLAGS.output, splits, queue))
      tasks = []
      for args in proc_feeder():
        tasks.append(pool.starmap_async(process_subreads, ([*args, queue],)))

        if main_counter['n_zmw_pass'] % 20 == 0:
          tasks = clear_tasks(tasks, main_counter)

      while tasks:
        time.sleep(0.5)
        tasks = clear_tasks(tasks, main_counter)

      # Cleanup multiprocessing.
      queue.put(['', 'kill'])
      tf_writer.get()
      manager.shutdown()
      pool.close()
      pool.join()
  # Write summary
  logging.info('Completed processing %s ZMWs.', main_counter['n_zmw_pass'])
  summary_name = 'training' if is_training else 'inference'
  dataset_summary = FLAGS.output.replace('.tfrecord.gz',
                                         f'.{summary_name}.json')
  # Remove @split from filenames
  dataset_summary = dataset_summary.replace('@split', 'summary')
  logging.info('Writing %s.', dataset_summary)
  make_dirs(dataset_summary)
  with open(dataset_summary, 'w') as summary_file:
    summary = dict(main_counter.items())
    summary.update(dc_config.to_dict())
    flag_list = [
        'subreads_to_ccs', 'ccs_fasta', 'truth_to_ccs', 'truth_bed',
        'truth_split'
    ]
    for flag in flag_list:
      summary[flag] = FLAGS[flag].value
    summary['version'] = dc_constants.__version__
    json_summary = json.dumps(summary, indent=True)
    summary_file.write(json_summary)

if __name__ == '__main__':
  logging.use_python_logging()
  app.run(main)
