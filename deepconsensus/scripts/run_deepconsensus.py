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
r"""Generates commands for and runs all stages of DeepConsensus.

1. Inference only.
2. Runs with direct runner (i.e. locally) only.

See the Quick Start in docs/quick_start.md for an example of how to run this
script.

"""

import os
import subprocess
from typing import Optional, List, TextIO

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_subreads_aligned', None,
    'Path to a BAM of input subreads, each aligned to its CCS sequence.')
flags.DEFINE_string(
    'input_subreads_unaligned', None,
    'Path to a BAM of input subreads that have the additional info from the '
    'sequencer like pulse widths. These can be and usually are the original '
    'unaligned BAMs from the sequencing run.')
flags.DEFINE_string(
    'input_ccs_fasta', None, 'Path to a FASTA of CCS sequences. '
    'These must be the sequences that --input_subreads_aligned are '
    'aligned against.')
flags.DEFINE_string(
    'output_directory', None,
    'Path specifying where to create a directory with the outputs from each '
    'step.')
flags.DEFINE_string(
    'checkpoint', None, 'Path to the model checkpoint to use. '
    'This should be the prefix of the model files, '
    'e.g. "checkpoint" when actual model files are at '
    'checkpoint.data-00000-of-00001 and checkpoint.index.')
flags.DEFINE_integer('min_quality', 20,
                     'Minimum quality filter for final output reads.')
flags.DEFINE_boolean(
    'dry_run', False,
    'Optional. If True, only prints out commands without executing them.')

EXAMPLE_WIDTH = 100

# pylint: disable=g-backslash-continuation


def create_all_commands(directories: List[str], input_subreads_aligned: str,
                        input_subreads_unaligned: str, input_ccs_fasta: str,
                        checkpoint: str, example_width: int,
                        min_quality: int) -> List[str]:
  """Create commands to run all stages of DeepConsensus.

  Args:
    directories: A list of paths to five directories, one for each stage.
    input_subreads_aligned: Path to aligned subreads BAM file.
    input_subreads_unaligned: Path to unaligned subreads BAM file.
    input_ccs_fasta: Path to CCS FASTA file.
    checkpoint: Path to checkpoint prefix.
    example_width: Integer width of examples. This should be consistent with the
      model checkpoint.
    min_quality: A quality filter to apply to the final reads.

  Returns:
    A list of commands, one for each stage of DeepConsensus.
  """

  command1 = f'python3 -m deepconsensus.preprocess.merge_datasets \
  --input_bam={input_subreads_aligned} \
  --input_unaligned_bam={input_subreads_unaligned} \
  --output_path={directories[0]} \
  --inference=true'

  command2 = f'python3 -m deepconsensus.preprocess.generate_input \
  --merged_datasets_path={directories[0]} \
  --output_path={directories[1]} \
  --input_ccs_fasta={input_ccs_fasta} \
  --inference=true'

  command3 = f'python3 -m deepconsensus.tf_examples.write_tf_examples \
  --preprocess_paths={directories[1]} \
  --output_path={directories[2]} \
  --preprocess_downsample=1 \
  --species=human \
  --inference=true \
  --example_width={example_width} \
  --max_passes=20 \
  --padded_len=120 \
  --subread_permutations=0'

  command4 = f'python3 -m deepconsensus.models.model_inference_with_beam \
  --dataset_path={directories[2]}/inference \
  --out_dir={directories[3]} \
  --checkpoint_path={checkpoint} \
  --inference=true \
  --max_passes=20'

  command5 = f'python3 -m deepconsensus.postprocess.stitch_predictions \
  --input_file={directories[3]}/predictions/*.tfrecords.gz \
  --output_path={directories[4]} \
  --example_width={example_width} \
  --inference=true \
  --min_quality={min_quality}'

  return [command1, command2, command3, command4, command5]


def run_command(command: str,
                dry_run: bool = True,
                log_file: Optional[TextIO] = None):
  """Run a given command in bash with nice logging, or optionally dry-run.

  Args:
    command: The command to execute.
    dry_run: False to actually execute the command, True to only print it.
    log_file: Optional file to output command, stdout, and stderr to. This is
      only used if dry_run is False. Regardless of whether this is True or
      False, the same information is also printed.

  Returns:
    Nothing. Prints and optionally executes with subprocess.

  Raises:
    RuntimeError: If the command fails to execute with a return code of 0.
  """

  if dry_run:
    print(f'\n***** DRY-RUN ONLY:*****\n{command}\n')
  else:
    banner = ('\n'
              '***** Running the command:*****\n'
              f'{command}\n'
              '*******************************\n')
    print(banner)
    if log_file is not None:
      print(banner, file=log_file)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        shell=True,
        executable='/bin/bash',
        universal_newlines=True) as proc:
      for line in proc.stdout:
        print(line, end='')
        if log_file is not None:
          print(line, end='', file=log_file)
      proc.communicate()  # This is needed to set proc.returncode.
      if proc.returncode != 0:
        raise RuntimeError(f'Command failed: \n{command}\n')


def run_deepconsensus(dry_run: bool, output_directory: str, **kwargs):
  """Run DeepConsensus with output directories and nice logging.

  Args:
    dry_run: True to just print commands, False to actually run them.
    output_directory: Directory to use for all outputs.
    **kwargs: These are passed to create_all_commands().
  """
  # Create an output directory for each stage.
  directories = [
      os.path.join(output_directory, '1_merge_datasets'),
      os.path.join(output_directory, '2_generate_input'),
      os.path.join(output_directory, '3_write_tf_examples'),
      os.path.join(output_directory, '4_model_inference_with_beam'),
      output_directory
  ]

  for subdirectory in directories:
    if dry_run:
      print(f'# DRY-RUN: mkdir {subdirectory}')
    else:
      if not os.path.isdir(subdirectory):
        os.makedirs(subdirectory)

  if not dry_run:
    log_file = open(
        os.path.join(output_directory, 'deepconsensus_log.txt'), 'w')
  else:
    log_file = None

  commands = create_all_commands(directories=directories, **kwargs)

  for command in commands:
    run_command(command, dry_run=dry_run, log_file=log_file)
  if log_file is not None:
    log_file.close()

  print('Outputs can be found at:', output_directory)


def main(_):
  """Main entry point."""

  run_deepconsensus(
      dry_run=FLAGS.dry_run,
      input_subreads_aligned=FLAGS.input_subreads_aligned,
      input_subreads_unaligned=FLAGS.input_subreads_unaligned,
      input_ccs_fasta=FLAGS.input_ccs_fasta,
      output_directory=FLAGS.output_directory,
      checkpoint=FLAGS.checkpoint,
      min_quality=FLAGS.min_quality,
      example_width=EXAMPLE_WIDTH)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'input_subreads_aligned', 'input_subreads_unaligned', 'input_ccs_fasta',
      'output_directory', 'checkpoint'
  ])
  app.run(main)
