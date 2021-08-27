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
r"""Script to calculate identity and gap-compressed identity of alignments.

The BAM file provided must be saved locally. This script won't work with files
on a remote filesystem. An index file is not required. You may see a warning
if the index file is not present, but the script will still run correctly.

Example usage:

python3 calculate_identity_metrics.py --bam_path reads.bam --output_dir /tmp
"""

import argparse
import collections
import os

import pysam

# Values used by pysam to encode different cigar ops.
EQUAL = 7
DIFF = 8
INS = 1
DEL = 2


def iterate_over_bam_pysam(bam_path: str, output_dir: str) -> None:
  """Writes out identity and gap-compressed identity of given BAM."""
  bam_file = pysam.AlignmentFile(bam_path, 'rb')
  entries = collections.defaultdict(int)
  bases = collections.defaultdict(int)
  output_path = os.path.join(output_dir, 'identity_metrics.csv')
  per_read_output_path = os.path.join(output_dir,
                                      'per_read_identity_metrics.csv')
  seen_names = set()
  with open(per_read_output_path, 'w') as f:
    f.write('read_name,identity,gap_compressed_identity\n')
    for read in bam_file:
      if read.is_secondary or read.is_supplementary:
        continue
      assert read.query_name not in seen_names
      seen_names.add(read.query_name)
      curr_entries = collections.defaultdict(int)
      curr_bases = collections.defaultdict(int)
      if not read.cigartuples:
        continue
      for op, oplen in read.cigartuples:
        entries[op] += 1
        bases[op] += oplen
        curr_entries[op] += 1
        curr_bases[op] += oplen
      curr_identity = 100.0 * curr_bases[EQUAL] / (
          curr_bases[EQUAL] + curr_bases[DIFF] + curr_bases[INS] +
          curr_bases[DEL])
      curr_gc_identity = 100.0 * curr_bases[EQUAL] / (
          curr_bases[EQUAL] + curr_bases[DIFF] + curr_entries[INS] +
          curr_entries[DEL])
      f.write(f'{read.query_name},{curr_identity},{curr_gc_identity}\n')

  identity = 100.0 * bases[EQUAL] / (
      bases[EQUAL] + bases[DIFF] + bases[INS] + bases[DEL])
  gcidentity = 100.0 * bases[EQUAL] / (
      bases[EQUAL] + bases[DIFF] + entries[INS] + entries[DEL])
  with open(output_path, 'w') as f:
    f.write('identity,gap_compressed_identity\n')
    f.write(f'{identity},{gcidentity}\n')
  print(f'Identity is: {identity}')
  print(f'Gap-compressed identity is: {gcidentity}')


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--bam_path', type=str, required=True, help='Path to input BAM file.')
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='Path to output CSV that will be written.')
  args = parser.parse_args()
  iterate_over_bam_pysam(bam_path=args.bam_path, output_dir=args.output_dir)


if __name__ == '__main__':
  main()
