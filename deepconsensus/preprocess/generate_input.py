r"""A pipeline for writing out DeepConsensusInput protos.

Subreads and labels, which are reads_pb2.Read protos, are processed and
joined with BED records to produce a DeepConsensusInput proto per molecule.
The DeepConsensusInput proto contains all information needed by models
downstream and can be transformed into tf.Example protos.

Example usage:

DATE=$(TZ=US/Pacific date "+%Y%m%d")
MERGED_DATASETS_PATH=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/merged_datasets/20200113
INPUT_BED=/cns/is-d/home/brain-genomics/deepvariant/dnanexus_pacbio/NDA/PacBioCCS/ecoli/refCoords/ecoliK12_pbi_August2018.m54316_180808_005743.ccs.bed
OUTPUT_PATH=/cns/is-d/home/brain-genomics/${USER}/deepconsensus/dc_input_datasets/${DATE}

time blaze run -c opt \
//learning/genomics/deepconsensus/preprocess:generate_input.par -- \
  --merged_datasets_path ${MERGED_DATASETS_PATH} \
  --input_bed ${INPUT_BED} \
  --output_path ${OUTPUT_PATH} \
  --flume_exec_mode=BORG \
  --flume_borg_user_name=${USER} \
  --flume_borg_accounting_charged_user_name=brain-genomics \
  --flume_batch_scheduler_strategy=RUN_SOON \
  --flume_use_batch_scheduler \
  --flume_worker_priority=100 \
  --flume_close_to_resources="/cns/is-d/home/brain-genomics" \
  --flume_backend=DAX \
  --flume_auto_retry=false \
  --flume_tmp_file_cells="is-d" \
  --flume_tmp_dir_group="brain-genomics" \
  --logtostderr \
  --flume_completion_email_address=${USER}@google.com
"""

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes

from deepconsensus.preprocess import beam_io
from deepconsensus.preprocess import generate_input_transforms
from deepconsensus.protos import deepconsensus_pb2
from google3.pipeline.flume.py import runner as flume_runner
from google3.third_party.nucleus.protos import reads_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('merged_datasets_path', None,
                    'Path containing merged subreads and label TFRecords.')
flags.DEFINE_string('input_bed', None, 'Input BED file.')
flags.DEFINE_string('input_ccs_fasta', None, 'Input CCS fasta file.')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')


def create_pipeline(merged_datasets_path, input_bed, input_ccs_fasta,
                    output_path):
  """Returns a pipeline for writing out DeepConsensusInput protos."""

  def pipeline(root):
    """Pipeline function for writing out DeepConsensusInput protos."""

    input_subreads = (
        root
        | 'read_subreads' >> beam.io.ReadFromTFRecord(
            file_pattern=os.path.join(merged_datasets_path, 'subreads/*'),
            coder=beam.coders.ProtoCoder(reads_pb2.Read),
            compression_type=CompressionTypes.GZIP)
        | 'reshuffle_subreads' >> beam.Reshuffle()  # to balance the shards
        | 'get_subread_molecule_name' >> beam.ParDo(
            generate_input_transforms.GetReadMoleculeNameDoFn()))

    input_labels = (
        root
        | 'read_labels' >> beam.io.ReadFromTFRecord(
            os.path.join(merged_datasets_path, 'labels/*'),
            coder=beam.coders.ProtoCoder(reads_pb2.Read),
            compression_type=CompressionTypes.GZIP)
        | 'reshuffle_labels' >> beam.Reshuffle()  # to balance the shards
        | 'get_label_molecule_name' >> beam.ParDo(
            generate_input_transforms.GetReadMoleculeNameDoFn()))

    aligned_subreads_and_label = (
        (input_subreads, input_labels)
        | 'group_subreads_and_labels' >> beam.CoGroupByKey()
        | 'expand_fields' >> beam.ParDo(
            generate_input_transforms.ExpandFieldsRemoveSoftClipsDoFn())
        | 'indent_read_start' >> beam.ParDo(
            generate_input_transforms.IndentReadsDoFn())
        | 'align_sequence' >> beam.ParDo(
            generate_input_transforms.AlignSequenceDoFn())
        | 'pad_read_end' >> beam.ParDo(generate_input_transforms.PadReadsDoFn())
        |
        'align_pw_ip' >> beam.ParDo(generate_input_transforms.AlignPwIpDoFn()))

    bed_records = (
        root
        | 'read_bed_records' >> beam_io.ReadBed(input_bed)
        | 'reshuffle_bed' >> beam.Reshuffle()
        | 'get_bed_molecule_name' >> beam.ParDo(
            generate_input_transforms.GetBedRecordMoleculeNameDoFn()))

    dc_inputs = ((aligned_subreads_and_label, bed_records)
                 | 'group_by_molecule' >> beam.CoGroupByKey()
                 | 'create_deepconsensus_input' >> beam.ParDo(
                     generate_input_transforms.CreateDeepConsensusInputDoFn()))

    if input_ccs_fasta:
      ccs_sequences = (
          root
          | 'read_fasta' >> beam_io.ReadFastaFile(input_ccs_fasta)
          | 'reshuffle_ccs_fasta' >> beam.Reshuffle()
          | 'get_ccs_molecule_name' >> beam.ParDo(
              generate_input_transforms.GetMoleculeNameFromSequenceName()))

      dc_inputs = (
          dc_inputs
          |
          'get_dc_molecule_name' >> beam.Map(lambda dc: (dc.molecule_name, dc)))

      dc_inputs = ((dc_inputs, ccs_sequences)
                   | 'group_dc_and_ccs' >> beam.CoGroupByKey()
                   | beam.ParDo(generate_input_transforms.AddCcsSequenceDoFn()))

    _ = (
        dc_inputs
        | 'write_deepconsensus_input' >> tfrecordio.WriteToTFRecord(
            os.path.join(output_path, 'deepconsensus/deepconsensus'),
            file_name_suffix='.tfrecords.gz',
            coder=beam.coders.ProtoCoder(deepconsensus_pb2.DeepConsensusInput),
            compression_type=CompressionTypes.GZIP))

  return pipeline


def main(argv):
  """Main entry point."""
  flume_runner.program_started()

  # We have to do flag validation in main rather than using
  # flags.mark_flags_as_required because beam workers don't set flags
  # appropriately.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.merged_datasets_path:
    raise app.UsageError('--merged_datasets_path must be specified.')
  if not FLAGS.input_bed:
    raise app.UsageError('--input_bed must be specified.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')

  runner = flume_runner.FlumeRunner()
  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.merged_datasets_path, FLAGS.input_bed,
                      FLAGS.input_ccs_fasta, FLAGS.output_path), options)


if __name__ == '__main__':
  app.run(main)
