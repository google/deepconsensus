r"""A pipeline for producing a unified set of read_pb2.Read protos for subreads and labels.

Example usage:

DATE=$(TZ=US/Pacific date "+%Y%m%d")
INPUT_BAM=/cns/is-d/home/brain-genomics/deepvariant/dnanexus_pacbio/NDA/PacBioCCS/ecoli/subreadsToCcs/m54316_180808_005743.subreadsToCcs.bam
INPUT_UNALIGNED_BAM=/cns/is-d/home/brain-genomics/deepvariant/dnanexus_pacbio/NDA/PacBioCCS/ecoli/subreads/m54316_180808_005743.subreads.bam
input_label_bam=/cns/is-d/home/brain-genomics/deepvariant/dnanexus_pacbio/NDA/PacBioCCS/ecoli/truth/m54316_180808_005743.truth_aligned_to_ccs.sorted.bam
input_label_fasta=/cns/is-d/home/brain-genomics/deepvariant/dnanexus_pacbio/NDA/PacBioCCS/ecoli/truth/m54316_180808_005743.truth.fasta.gz
OUTPUT_PATH=/cns/is-d/home/brain-genomics/${USER}/deepconsensus/merged_datasets/${DATE}

time blaze run -c opt \
//learning/genomics/deepconsensus/preprocess:merge_datasets.par -- \
  --input_bam ${INPUT_BAM} \
  --input_unaligned_bam ${INPUT_UNALIGNED_BAM} \
  --input_label_bam ${input_label_bam} \
  --input_label_fasta ${input_label_fasta} \
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
from deepconsensus.preprocess import merge_datasets_transforms
from google3.pipeline.flume.py import runner as flume_runner
from google3.third_party.nucleus.protos import reads_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('input_bam', None,
                    'Input BAM file - subreads aligned to ccs.')
flags.DEFINE_string('input_unaligned_bam', None,
                    'Input unaligned subreads BAM.')
flags.DEFINE_string('input_label_bam', None,
                    'Input BAM file - labels aligned to ccs.')
flags.DEFINE_string('input_label_fasta', None,
                    'Input FASTA file of label sequences.')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')


def create_pipeline(input_bam, input_unaligned_bam, input_label_bam,
                    input_label_fasta, output_path):
  """Returns a pipeline for merging input datasets for subreads and label."""

  def pipeline(root):
    """Pipeline function for merging input datasets."""
    input_subreads = (
        root
        | 'read_reads' >> beam_io.ReadSam(input_bam)
        | 'reshuffle_contigs' >> beam.Reshuffle()  # to balance the shards.
        | 'remove_reads_missing_sequence' >> beam.ParDo(
            merge_datasets_transforms.RemoveReadsMissingSequenceDoFn())
        | 'filter_incorrectly_mapped_subreads' >> beam.ParDo(
            merge_datasets_transforms.RemoveIncorrectlyMappedReadsDoFn())
        | 'get_read_name' >> beam.ParDo(
            merge_datasets_transforms.GetReadNameDoFn()))

    input_unaligned_subreads = (
        root
        | 'read_unaligned_reads' >> beam_io.ReadSam(
            input_unaligned_bam, parse_aux_fields=True)
        | 'reshuffle_unaligned_reads' >> beam.Reshuffle()
        | 'get_unaligned_read_name' >> beam.ParDo(
            merge_datasets_transforms.GetReadNameDoFn()))

    _ = ((input_subreads, input_unaligned_subreads)
         | 'group_by_read_name' >> beam.CoGroupByKey()
         | 'merge_read_protos' >> beam.ParDo(
             merge_datasets_transforms.MergeSubreadsDoFn())
         | 'write_merged_subreads' >> tfrecordio.WriteToTFRecord(
             os.path.join(output_path, 'subreads/subreads'),
             file_name_suffix='.tfrecords.gz',
             coder=beam.coders.ProtoCoder(reads_pb2.Read),
             compression_type=CompressionTypes.GZIP))

    label_reads = (
        root
        | 'read_label_from_bam' >> beam_io.ReadSam(input_label_bam)
        | 'reshuffle_label' >> beam.Reshuffle()  # to balance the shards
        | 'filter_incorrectly_mapped_label_reads' >> beam.ParDo(
            merge_datasets_transforms.RemoveIncorrectlyMappedReadsDoFn())
        | 'get_read_fragment_name' >> beam.ParDo(
            merge_datasets_transforms.GetReadNameDoFn()))

    label_sequences = (
        root
        | 'read_label_from_fasta' >> beam_io.ReadFastaFile(input_label_fasta)
        | 'reshuffle_label_bases' >> beam.Reshuffle())  # to balance the shards

    _ = ((label_reads, label_sequences)
         | 'group_label_and_bases_by_molecule' >> beam.CoGroupByKey()
         | 'add_sequence_to_label_reads' >> beam.ParDo(
             merge_datasets_transforms.MergeLabelsDoFn())
         | 'write_merged_label' >> tfrecordio.WriteToTFRecord(
             os.path.join(output_path, 'labels/labels'),
             file_name_suffix='.tfrecords.gz',
             coder=beam.coders.ProtoCoder(reads_pb2.Read),
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
  if not FLAGS.input_bam:
    raise app.UsageError('--input_bam must be specified.')
  if not FLAGS.input_unaligned_bam:
    raise app.UsageError('--input_unaligned_bam must be specified.')
  if not FLAGS.input_label_bam:
    raise app.UsageError('--input_label_bam must be specified.')
  if not FLAGS.input_label_fasta:
    raise app.UsageError('--input_label_fasta must be specified.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')

  runner = flume_runner.FlumeRunner()
  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.input_bam, FLAGS.input_unaligned_bam,
                      FLAGS.input_label_bam, FLAGS.input_label_fasta,
                      FLAGS.output_path), options)


if __name__ == '__main__':
  app.run(main)
