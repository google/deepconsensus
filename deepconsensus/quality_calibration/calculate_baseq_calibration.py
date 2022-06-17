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
r"""Calculate the empirical quality against the predicted quality of reads.

The input bam file can be collected from bam_concordance/prediction.bam which
is the output of run_deepconsensus.sh

Example run:

deepconsensus calculate_baseq_calibration \
--bam prediction.bam \
--ref chm13v2.0_noY.fa \
--region chr20 \
--output_csv CHM13_m64062_190803_042216_chr20_exp36943996_wu_1_c50.bq_bins.csv
--cpus 20
"""

import collections
import functools
import multiprocessing
import multiprocessing.pool
import time
from typing import Dict, List, Any

from absl import flags
from absl import logging
import pandas as pd
import pysam
import tensorflow as tf
from absl import app


AsyncResult = multiprocessing.pool.AsyncResult
MAX_BASEQ = 100

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'bam',
    default=None,
    help='Input BAM containing reads aligned to reference alignment.')
flags.DEFINE_string(
    'ref', default=None, help='Input FASTA file of the reference/assembly.')
flags.DEFINE_string(
    'region',
    default=None,
    help='A region defined as contig:start-stop. [chr20:1000-2000]')
flags.DEFINE_string(
    'output_csv',
    default=None,
    help='Path to a CSV output filename. example: /path/OUTPUT.csv')
flags.DEFINE_integer(
    'cpus',
    default=multiprocessing.cpu_count(),
    help='Number of worker processes to use.',
    short_name='j')
flags.DEFINE_integer(
    'interval_length',
    default=1000,
    help='Region interval length for splitting into smaller chunks.')
flags.DEFINE_integer(
    'min_mapq',
    default=60,
    help='Minimum mapping quality for read.'
    'Reads below min_mapq will be ignored from calculation.')


def register_required_flags():
  flags.mark_flags_as_required(['bam', 'ref', 'output_csv'])


class RegionRecord:
  """Represents a genomics region.

    : contig - Name of a contig
    : start - start position
    : end - end position
  """

  def __init__(self, contig: str, start: int, stop: int):
    self.contig = contig
    self.start = start
    self.stop = stop

  def __str__(self):
    return '[REGION: Contig= %s, Start= %d, Stop= %d]' % (self.contig,
                                                          self.start, self.stop)


def process_region_string(region_string: str, fasta_file: str) -> RegionRecord:
  """Takes region string in contig:start-stop and returns list [contig, start, stop]."""
  if ':' in region_string:
    if len(region_string.split(':')) != 2:
      raise ValueError('Malformed region string %s' % region_string)
    contig, start_stop = region_string.split(':')

    if len(start_stop.split('-')) != 2:
      raise ValueError('Malformed region string %s' % region_string)

    start, stop = start_stop.split('-')

    try:
      region_record = RegionRecord(contig, int(start), int(stop))
    except ValueError:
      print('Malformed region string %s' % region_string)

    if region_record.start > region_record.stop:
      raise ValueError('Malformed region string %s' % region_string)
  else:
    # expected that the passed value is a contig
    contig = region_string
    fasta_reader = pysam.FastaFile(fasta_file)
    fasta_contigs = fasta_reader.references
    if contig not in fasta_contigs:
      raise ValueError('Contig %s not found in fasta' % contig)
    contig_len = fasta_reader.get_reference_length(contig)
    region_record = RegionRecord(contig, 0, int(contig_len))
    fasta_reader.close()

  return region_record


def split_regions_in_intervals(regions: List[RegionRecord],
                               region_length: int) -> List[RegionRecord]:
  """Splits each region into intervals of region length.

  Args:
    regions: List of regions.
    region_length: Length of intervals

  Returns:
    A list of regions of maximum length of region_length.
  """
  all_intervals = []
  for region in regions:
    for pos in range(region.start, region.stop, region_length):
      interval_start = max(region.start, pos)
      interval_end = min(region.stop, pos + region_length)
      interval_record = RegionRecord(region.contig, interval_start,
                                     interval_end)
      all_intervals.append(interval_record)
  return all_intervals


def get_contig_regions(bam_file: str, fasta_file: str, region: str,
                       interval_length: int) -> List[RegionRecord]:
  """Creates a list of regions for processing.

  Reads contig names from bam and fasta file and creates a list of regions
  for processing.

  Args:
    bam_file: Path to an alignment bam file.
    fasta_file: Path to a fasta file.
    region: A region string in the format contig:start-end.
    interval_length: The length of intervals.

  Returns:
    A list of region intervals.
  """
  bam_reader = pysam.AlignmentFile(bam_file)
  bam_contigs = bam_reader.references

  fasta_reader = pysam.FastaFile(fasta_file)
  fasta_contigs = fasta_reader.references

  bam_fasta_common_contigs = list(set(fasta_contigs) & set(bam_contigs))
  regions_to_process = []

  if region:  # if region parameter has been set
    if ',' in region:
      contigs = region.split(',')
      for contig in contigs:
        region_record = process_region_string(contig, fasta_file)
        if region_record.contig not in bam_fasta_common_contigs:
          raise ValueError('Contig %s not found in BAM or FASTA file.' %
                           region_record.contig)
        regions_to_process.append(region_record)
    else:
      region_record = process_region_string(region, fasta_file)
      if region_record.contig not in bam_fasta_common_contigs:
        raise ValueError('Contig %s not found in BAM or FASTA file.' %
                         region_record.contig)
      regions_to_process.append(region_record)
  else:  # create regions from the common contigs
    for contig in bam_fasta_common_contigs:
      region_record = RegionRecord(contig, 0,
                                   fasta_reader.get_reference_length(contig))
      regions_to_process.append(region_record)

  region_intervals = split_regions_in_intervals(regions_to_process,
                                                interval_length)

  bam_reader.close()
  fasta_reader.close()

  return region_intervals


def create_processes(bam_file: str, fasta_file: str,
                     all_intervals: List[RegionRecord], total_threads: int,
                     min_mapq: int) ->...:
  """Argument generator for launching processes in parallel."""

  def process_feeder():
    for thread in range(0, total_threads):
      process_intervals = [
          r for i, r in enumerate(all_intervals) if i % total_threads == thread
      ]
      yield (bam_file, fasta_file, process_intervals, min_mapq)

  return process_feeder


def trace_exception(f) ->...:
  """Decorator to catch errors run in multiprocessing processes."""

  @functools.wraps(f)
  def wrap(*args, **kwargs):
    try:
      result = f(*args, **kwargs)
      return result
    except:  # pylint: disable=bare-except
      logging.exception('Error in function %s.', f.__name__)
      raise Exception('Error in worker process') from None

  return wrap


def clear_tasks(tasks: List[Any], global_stats: List[Dict[str,
                                                          int]]) -> List[Any]:
  """Clear successful tasks and log result."""
  for task in tasks:
    if task.ready():
      if task.successful():
        # Fetch task results and integrate into main counter
        match_mismatch_count = task.get()[0]
        for i in range(0, MAX_BASEQ):
          global_stats[i]['M'] += match_mismatch_count[i]['M']
          global_stats[i]['X'] += match_mismatch_count[i]['X']
        tasks.remove(task)
      else:
        raise Exception('A worker process failed.')
  return tasks


def get_quality_calibration_stats(reads: List[pysam.AlignedSegment],
                                  ref_sequence: str,
                                  region_interval: RegionRecord,
                                  min_mapq: int) -> List[Dict[str, int]]:
  """Iterate over reads and calculate quality scores."""
  match_mismatch_count = [{'M': 0, 'X': 0} for _ in range(0, MAX_BASEQ)]

  for read in reads:
    if read.is_duplicate or read.is_qcfail or read.is_secondary or read.is_unmapped:
      continue

    if read.is_supplementary or read.mapping_quality < min_mapq:
      continue

    current_ref_pos = read.reference_start
    current_read_index = 0

    # iterate over the read to find the maximum insert size
    # we observe at any position within the window.
    for cigar_op, cigar_len in read.cigartuples:
      # we have skipped past the window, no need to process this read anymore
      if current_ref_pos > region_interval.stop:
        break
      # if it's a match then only move forward, matches don't need padding
      if cigar_op in [pysam.CMATCH, pysam.CDIFF, pysam.CEQUAL]:
        for _ in range(0, cigar_len):
          # the base is within the window
          if region_interval.start <= current_ref_pos <= region_interval.stop:
            region_index = current_ref_pos - region_interval.start
            ref_base = ref_sequence[region_index].upper()
            read_base = read.query_sequence[current_read_index].upper()
            read_base_quality = read.query_qualities[current_read_index]
            if ref_base.upper() in ['A', 'C', 'G', 'T']:
              if ref_base != read_base:
                match_mismatch_count[read_base_quality]['X'] += 1
              else:
                match_mismatch_count[read_base_quality]['M'] += 1

          current_read_index += 1
          current_ref_pos += 1
      # if it's an insert then we record it
      elif cigar_op in [pysam.CSOFT_CLIP, pysam.CINS]:
        for _ in range(0, cigar_len):
          # the base is within the window
          if region_interval.start <= current_ref_pos <= region_interval.stop:
            read_base = read.query_sequence[current_read_index].upper()
            read_base_quality = read.query_qualities[current_read_index]
            match_mismatch_count[read_base_quality]['X'] += 1

          current_read_index += 1
      # If it's a delete then move forward.
      elif cigar_op in [pysam.CREF_SKIP, pysam.CDEL]:
        # skip deleted bases
        current_ref_pos += cigar_len

  return match_mismatch_count


def calculate_quality_calibration(bam_file: str, fasta_file: str,
                                  process_intervals: List[RegionRecord],
                                  min_mapq: int) -> List[Dict[str, int]]:
  """Calculate quality calibration of reads."""
  thread_counter = collections.Counter()
  thread_counter['n_examples'] += len(process_intervals)
  bam_reader = pysam.AlignmentFile(bam_file)
  fasta_reader = pysam.FastaFile(fasta_file)
  main_dict = [{'M': 0, 'X': 0} for _ in range(0, MAX_BASEQ)]
  for interval_region in process_intervals:
    # get the sequence from the fasta file
    reference_sequence = fasta_reader.fetch(interval_region.contig,
                                            interval_region.start,
                                            interval_region.stop + 5)
    # get the reads from bam
    read_set = bam_reader.fetch(interval_region.contig, interval_region.start,
                                interval_region.stop)
    # create the example of the region
    match_mismatch_count = get_quality_calibration_stats(
        read_set, reference_sequence, interval_region, min_mapq)
    for i in range(0, MAX_BASEQ):
      main_dict[i]['M'] += match_mismatch_count[i]['M']
      main_dict[i]['X'] += match_mismatch_count[i]['X']
  bam_reader.close()
  fasta_reader.close()

  return main_dict


def save_csv(df: pd.DataFrame, path: str):
  """Save pandas dataframe as a CSV."""
  with tf.io.gfile.GFile(path, 'w') as csv_out:
    df.to_csv(csv_out, sep=',', index=False)


def main(unused_argv) -> None:
  if FLAGS.cpus == 0:
    raise ValueError('Must set cpus to >=1 for processing.')
  # get all intervals
  all_intervals = get_contig_regions(FLAGS.bam, FLAGS.ref, FLAGS.region,
                                     FLAGS.interval_length)

  manager = multiprocessing.Manager()

  proc_feeder = create_processes(FLAGS.bam, FLAGS.ref, all_intervals,
                                 FLAGS.cpus, FLAGS.min_mapq)
  global_match_mismatch_stat = [{'M': 0, 'X': 0} for _ in range(0, MAX_BASEQ)]

  logging.info('Processing in parallel using %s cores', FLAGS.cpus)
  with multiprocessing.Pool(FLAGS.cpus) as pool:
    tasks = []
    for args in proc_feeder():
      tasks.append(
          pool.starmap_async(calculate_quality_calibration, ([*args],)))
    while tasks:
      time.sleep(0.5)
      tasks = clear_tasks(tasks, global_match_mismatch_stat)

    # Cleanup multiprocessing.
    manager.shutdown()
    pool.close()
    pool.join()

  base_quality_dataframe = pd.DataFrame(
      columns=['baseq', 'total_match', 'total_mismatch'])
  for baseq in range(0, MAX_BASEQ):
    base_quality_dataframe = base_quality_dataframe.append(
        {
            'baseq': str(baseq),
            'total_match': str(global_match_mismatch_stat[baseq]['M']),
            'total_mismatch': str(global_match_mismatch_stat[baseq]['X'])
        },
        ignore_index=True)
  save_csv(base_quality_dataframe, FLAGS.output_csv)
  print('Processing complete.')


if __name__ == '__main__':
  logging.use_python_logging()
  app.run(main)
