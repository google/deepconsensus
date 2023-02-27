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
"""Tests for filter_reads."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam

from deepconsensus.quality_calibration import filter_reads
from deepconsensus.utils import test_utils
from absl import app



FLAGS = flags.FLAGS


class FilterReadsE2Etest(parameterized.TestCase):
  """E2E tests for filter reads."""

  def get_all_reads_from_file(self, filename):
    """Returns all reads from a fastq file.

    Args:
      filename: Path to input fastq file.

    Returns:
        A list of reads in fastq file.
    """
    input_file = pysam.FastxFile(filename)
    with input_file as input_reads:
      all_reads = list(input_reads)
    return all_reads

  @parameterized.parameters(
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919_q0_chr20_100reads.q0.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919_q0_chr20_100reads.fq.gz"
              ),
              "output_fastq": "output.q0.fq",
              "quality_threshold": 0,
          },
      ),
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919_q0_chr20_100reads.q10.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919_q0_chr20_100reads.fq.gz"
              ),
              "output_fastq": "output.q10.fq",
              "quality_threshold": 10,
          },
      ),
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919_q0_chr20_100reads.q20.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919_q0_chr20_100reads.fq.gz"
              ),
              "output_fastq": "output.q20.fq",
              "quality_threshold": 20,
          },
      ),
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919_q0_chr20_100reads.q30.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919_q0_chr20_100reads.fq.gz"
              ),
              "output_fastq": "output.q30.fq",
              "quality_threshold": 30,
          },
      ),
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919_q0_chr20_100reads.q40.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919_q0_chr20_100reads.fq.gz"
              ),
              "output_fastq": "output.q40.fq",
              "quality_threshold": 40,
          },
      ),
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919_q0_chr20_100reads.q50.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919_q0_chr20_100reads.fq.gz"
              ),
              "output_fastq": "output.q50.fq",
              "quality_threshold": 50,
          },
      ),
      dict(
          expected_output_fastq=(
              "filter_fastq/m64062_190806_063919-chr20.dc.small.q30.fq.gz"
          ),
          flag_values={
              "input_seq": (
                  "filter_fastq/m64062_190806_063919-chr20.dc.small.bam"
              ),
              "output_fastq": "output.bam.q30.fq",
              "quality_threshold": 30,
          },
      ),
  )
  @flagsaver.flagsaver
  def test_filter_fastq_golden(self, expected_output_fastq, flag_values):
    """Test filter_fastq method."""
    # Set flag values.
    FLAGS.input_seq = test_utils.deepconsensus_testdata(
        flag_values["input_seq"]
    )
    # Create a tmp dir.
    tmp_dir = self.create_tempdir()
    output = os.path.join(tmp_dir, flag_values["output_fastq"])
    FLAGS.output_fastq = output
    FLAGS.quality_threshold = flag_values["quality_threshold"]
    filter_reads.main([])
    expected_fastq = test_utils.deepconsensus_testdata(expected_output_fastq)
    expected_read_list = self.get_all_reads_from_file(expected_fastq)
    read_list = self.get_all_reads_from_file(output)
    for filtered_read, expected_read in zip(read_list, expected_read_list):
      self.assertEqual(filtered_read.name, expected_read.name)
      self.assertEqual(filtered_read.quality, expected_read.quality)
      self.assertEqual(filtered_read.sequence, expected_read.sequence)


if __name__ == "__main__":
  absltest.main()
