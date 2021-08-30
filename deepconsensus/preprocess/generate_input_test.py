# Copyright (c) 2021, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Tests for deepconsensus.preprocess.generate_input."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam

from deepconsensus.preprocess import generate_input
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils.test_utils import deepconsensus_testdata
from nucleus.io import sharded_file_utils
from nucleus.io import tfrecord
from nucleus.protos import bed_pb2


class GenerateInputTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ecoli', 1, True),
      ('ecoli', 1, False),
      ('human', 2, True),
      ('human', 2, False),
  )
  def test_end_to_end(self, species, expected_dc_input_count, inference):
    """Tests the full pipeline for generating DeepConsensusInput protos."""

    output_path = 'inference_output' if inference else 'output'
    merged_datasets_path = deepconsensus_testdata(f'{species}/{output_path}')
    temp_dir = self.create_tempdir().full_path
    input_ccs_fasta = deepconsensus_testdata(f'{species}/{species}.ccs.fasta')
    if inference:
      input_bed = None
    else:
      input_bed = deepconsensus_testdata(f'{species}/{species}.refCoords.bed')
    runner = beam.runners.DirectRunner()
    pipeline = generate_input.create_pipeline(
        merged_datasets_path=merged_datasets_path,
        input_bed=input_bed,
        input_ccs_fasta=input_ccs_fasta,
        output_path=temp_dir,
        inference=inference)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    runner.run(pipeline, options)

    dc_input_files = sharded_file_utils.glob_list_sharded_file_patterns(
        os.path.join(temp_dir, 'deepconsensus/deepconsensus*.tfrecords.gz'))
    dc_input_pattern = os.path.join(
        temp_dir,
        'deepconsensus/deepconsensus@%d.tfrecords.gz' % len(dc_input_files))
    reader = tfrecord.read_tfrecords(
        dc_input_pattern, proto=deepconsensus_pb2.DeepConsensusInput)

    # Sanity checks for the DeepConsensusInput protos written out.
    dc_input_count = 0
    for dc_input in reader:
      seq_len = len(dc_input.subreads[0].bases)
      self.assertNotEmpty(dc_input.subreads)
      self.assertNotEmpty(dc_input.molecule_name)
      self.assertGreaterEqual(dc_input.molecule_start, 0)
      self.assertLen(dc_input.sn, 4)
      self.assertNotEmpty(dc_input.ccs_sequence)
      for subread in dc_input.subreads:
        self.assertNotEmpty(subread.molecule_name)
        self.assertLen(subread.bases, seq_len)
        self.assertLen(subread.expanded_cigar, seq_len)
        self.assertLen(subread.pw, seq_len)
        self.assertLen(subread.ip, seq_len)
      if inference:
        self.assertEqual(dc_input.strand, bed_pb2.BedRecord.Strand.NO_STRAND)
      else:
        self.assertNotEqual(dc_input.strand, bed_pb2.BedRecord.Strand.NO_STRAND)
        self.assertNotEmpty(dc_input.chrom_name)
        self.assertGreaterEqual(dc_input.chrom_start, 0)
        self.assertGreater(dc_input.chrom_end, 0)
        self.assertLen(dc_input.label.bases, seq_len)
        self.assertLen(dc_input.label.expanded_cigar, seq_len)
      dc_input_count += 1

    self.assertEqual(dc_input_count, expected_dc_input_count)


if __name__ == '__main__':
  absltest.main()
