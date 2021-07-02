"""Tests for deepconsensus .preprocess.generate_input."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam

from deepconsensus.preprocess import generate_input
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils.test_utils import deepconsensus_testdata
from google3.pipeline.flume.py import runner as flume_runner
from google3.third_party.nucleus.io import sharded_file_utils
from google3.third_party.nucleus.io import tfrecord
from google3.third_party.nucleus.protos import bed_pb2


class GenerateInputTest(parameterized.TestCase):

  @parameterized.parameters(('ecoli', 1), ('human', 2))
  def test_end_to_end(self, species, expected_dc_input_count):
    """Tests the full pipeline for generating DeepConsensusInput protos."""

    merged_datasets_path = deepconsensus_testdata(f'{species}/output')
    input_bed = deepconsensus_testdata(f'{species}/{species}.refCoords.bed')
    temp_dir = self.create_tempdir().full_path
    input_ccs_fasta = deepconsensus_testdata(f'{species}/{species}.ccs.fasta')
    runner = flume_runner.FlumeRunner()
    pipeline = generate_input.create_pipeline(
        merged_datasets_path=merged_datasets_path,
        input_bed=input_bed,
        input_ccs_fasta=input_ccs_fasta,
        output_path=temp_dir)
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
      self.assertNotEqual(dc_input.strand, bed_pb2.BedRecord.Strand.NO_STRAND)
      self.assertNotEmpty(dc_input.subreads)
      self.assertNotEmpty(dc_input.molecule_name)
      self.assertNotEmpty(dc_input.chrom_name)
      self.assertGreaterEqual(dc_input.molecule_start, 0)
      self.assertGreaterEqual(dc_input.chrom_start, 0)
      self.assertGreater(dc_input.chrom_end, 0)
      self.assertLen(dc_input.sn, 4)
      self.assertNotEmpty(dc_input.ccs_sequence)

      seq_len = len(dc_input.subreads[0].bases)
      self.assertLen(dc_input.label.bases, seq_len)
      self.assertLen(dc_input.label.expanded_cigar, seq_len)
      for subread in dc_input.subreads:
        self.assertNotEmpty(subread.molecule_name)
        self.assertLen(subread.bases, seq_len)
        self.assertLen(subread.expanded_cigar, seq_len)
        self.assertLen(subread.pw, seq_len)
        self.assertLen(subread.ip, seq_len)

      dc_input_count += 1

    self.assertEqual(dc_input_count, expected_dc_input_count)


if __name__ == '__main__':
  absltest.main()
