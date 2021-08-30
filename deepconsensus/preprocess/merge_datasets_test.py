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
"""Tests for deepconsensus.merge_datasets."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam

from deepconsensus.preprocess import merge_datasets
from deepconsensus.utils.test_utils import deepconsensus_testdata
from nucleus.io import sharded_file_utils
from nucleus.io import tfrecord
from nucleus.protos import reads_pb2
from nucleus.util import struct_utils


class MergeDatasetsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ecoli', 1, True),
      ('ecoli', 1, False),
      ('human', 2, True),
      ('human', 2, False),
  )
  def test_end_to_end(self, species, expected_label_count, inference):
    """Tests that full pipeline runs without errors and produces outputs.

    Check that:

    * Merged subread reads_pb2.Read protos exist and have sequence, cigar, pulse
    width (pw), interpulse distance (ip), and signal to noise (sn) values of
    correct sizes.

    * Merged label reads_pb2.Read protos exist and have sequence.

    The behavior of each DoFn is not tested here. See
    //learning/genomics/deepconsensus/preprocess/merge_datasets_transforms_test.py
    for tests corresponding to each DoFn.

    Args:
      species: string used to complete data paths. Either 'ecoli' or 'human'.
      expected_label_count: number of labels we expect in the output.
      inference: whether to run in inference or training mode.
    """

    input_bam = deepconsensus_testdata(f'{species}/{species}.subreadsToCcs.bam')
    input_unaligned_bam = deepconsensus_testdata(
        f'{species}/{species}.subreads.bam')
    if not inference:
      input_label_bam = deepconsensus_testdata(
          f'{species}/{species}.truthToCcs.bam')
      input_label_fasta = deepconsensus_testdata(
          f'{species}/{species}.truth.fasta')
    else:
      input_label_bam = ''
      input_label_fasta = ''

    temp_dir = self.create_tempdir().full_path

    runner = beam.runners.DirectRunner()
    pipeline = merge_datasets.create_pipeline(
        input_bam=input_bam,
        input_unaligned_bam=input_unaligned_bam,
        input_label_bam=input_label_bam,
        input_label_fasta=input_label_fasta,
        output_path=temp_dir,
        inference=inference)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    runner.run(pipeline, options)

    # Sanity checks for the merged subreads written out.
    subread_files = sharded_file_utils.glob_list_sharded_file_patterns(
        os.path.join(temp_dir, 'subreads/subreads*.tfrecords.gz'))
    subread_pattern = os.path.join(
        temp_dir, 'subreads/subreads@%d.tfrecords.gz' % len(subread_files))
    reader = tfrecord.read_tfrecords(subread_pattern, proto=reads_pb2.Read)

    subread_count = 0
    for subread in reader:
      self.assertNotEmpty(subread.aligned_sequence)
      self.assertNotEmpty(subread.alignment.cigar)
      self.assertNotEmpty(subread.info['pw'].values)
      self.assertNotEmpty(subread.info['ip'].values)
      self.assertNotEmpty(subread.info['sn'].values)

      seq_len = len(subread.aligned_sequence)
      self.assertLen(struct_utils.get_int_field(subread.info, 'pw'), seq_len)
      self.assertLen(struct_utils.get_int_field(subread.info, 'ip'), seq_len)
      self.assertLen(struct_utils.get_int_field(subread.info, 'sn'), 4)
      subread_count += 1
    self.assertGreater(subread_count, 0)

    # Sanity checks for the merged labels written out.
    if not inference:
      label_files = sharded_file_utils.glob_list_sharded_file_patterns(
          os.path.join(temp_dir, 'labels/labels*.tfrecords.gz'))
      label_pattern = os.path.join(
          temp_dir, 'labels/labels@%d.tfrecords.gz' % len(label_files))
      reader = tfrecord.read_tfrecords(label_pattern, proto=reads_pb2.Read)

      label_count = 0
      for label in reader:
        self.assertNotEmpty(label.aligned_sequence)
        label_count += 1
      self.assertEqual(label_count, expected_label_count)
    else:
      label_files = sharded_file_utils.glob_list_sharded_file_patterns(
          os.path.join(temp_dir, 'labels/labels*.tfrecords.gz'))
      self.assertEmpty(label_files)


if __name__ == '__main__':
  absltest.main()
