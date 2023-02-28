#!/usr/bin/env python
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
"""Filter reads based on the base qualities."""

from collections.abc import Sequence
import math
from absl import app
from absl import flags
from absl import logging
import pysam

_INPUT_SEQ = flags.DEFINE_string(
    'input_seq',
    short_name='i',
    default=None,
    help='Path to input fastq or bam file.',
)

_OUTPUT_FASTQ = flags.DEFINE_string(
    'output_fastq',
    short_name='o',
    default=None,
    help='Path to output fastq file.',
)

_QUALITY_THRESHOLD = flags.DEFINE_integer(
    'quality_threshold',
    short_name='q',
    default=None,
    help=(
        'A quality threshold value applied per read. Reads with read quality'
        ' below this will be filtered out.'
    ),
)


def register_required_flags():
  flags.mark_flags_as_required(['input_seq', 'output_fastq',
                                'quality_threshold'])


def avg_phred(base_qualities: Sequence[float]) -> float:
  """Get the average phred quality given base qualities of a read.

  Args:
     base_qualities: A list containing the base qualities of a read.

  Returns:
     The average error rate of the read quality list.
  """
  # Making sure we have the base quality.
  if not base_qualities:
    return 0
  return -10 * math.log10(
      sum([10**(i / -10) for i in base_qualities]) / int(len(base_qualities)))


def filter_bam_or_fastq_by_quality(
    input_seq: str, output_fastq: str, quality_threshold: int
):
  """Filter reads of a fastq file based on base quality.

  Args:
    input_seq: Path to a fastq or bam.
    output_fastq: Path to a output fastq file.
    quality_threshold: A threshold value.
  """
  output_fastq = open(output_fastq, mode='w')

  total_reads = 0
  total_reads_above_q = 0
  if input_seq.endswith('.bam'):
    input_file = pysam.AlignmentFile(input_seq, check_sq=False)
  else:
    input_file = pysam.FastxFile(input_seq)
  with input_file as input_reads:
    for read in input_reads:
      total_reads += 1
      # Round the phred score to ensure expected behavior. Without rounding, a
      # read with all base qualities equal to 10 will have an average phred of
      # 9.99999 due to python floating point precision. Such a read would get
      # filtered out if min_quality is 10.
      if isinstance(read, pysam.AlignedSegment):
        # Get avg phred from bam read.
        phred = avg_phred(read.query_qualities)
      else:
        # Get avg phred from fastq quality scores.
        phred = round(avg_phred(read.get_quality_array()), 5)
      if phred >= quality_threshold:
        total_reads_above_q += 1
        if isinstance(read, pysam.AlignedSegment):
          record = '\n'.join(
              ['@' + read.qname, read.query_sequence, '+', read.qual]) + '\n'
          output_fastq.write(record)
        else:
          output_fastq.write(str(read) + '\n')

  output_fastq.close()
  # Output from this script is piped directly to an output file.
  logging.info('TOTAL READS IN INPUT: %d', total_reads)
  logging.info('TOTAL READS IN OUTPUT: %d', total_reads_above_q)
  logging.info(
      'TOTAL FILTERED READS: %d',
      total_reads - total_reads_above_q,
  )


def main(unused_argv) -> None:
  filter_bam_or_fastq_by_quality(
      _INPUT_SEQ.value,
      _OUTPUT_FASTQ.value,
      _QUALITY_THRESHOLD.value,
  )


if __name__ == '__main__':
  logging.use_python_logging()
  register_required_flags()
  app.run(main)
