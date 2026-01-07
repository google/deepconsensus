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
r"""Visualize DeepConsensus TFRecord examples in a human-readable format.

Example usage:
blaze run -c opt //learning/genomics/deepconsensus/utils:show_examples -- \
  --file_path=/path/to/tfrecord.gz \
  --alsologtostderr \
  --num_records=10
"""

from collections.abc import Callable, Sequence

from absl import app
from absl import flags
import colorama
import immutabledict
import numpy as np
import tensorflow.compat.v2 as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_configs
from deepconsensus.preprocess import pre_lib
from deepconsensus.utils import dc_constants

_FILE_PATH = flags.DEFINE_string(
    'file_path',
    None,
    'Path to tfrecord.gz file to visualize.',
    required=True,
)
_NUM_RECORDS = flags.DEFINE_integer(
    'num_records',
    10,
    'Number of records to visualize.',
)
_NAMES = flags.DEFINE_list(
    'names',
    None,
    'Comma-separated list of example names to visualize.',
)
_SPACE_EVERY_N = flags.DEFINE_integer(
    'space_every_n',
    10,
    'If > 0, adds a space every n bases for improved readability.',
)
_COLOR = flags.DEFINE_bool(
    'color',
    True,
    'If True, use color in output.',
)

_DNA_COLORS = immutabledict.immutabledict({
    'A': str(colorama.Fore.GREEN),
    'T': str(colorama.Fore.RED),
    'C': str(colorama.Fore.BLUE),
    'G': str(colorama.Fore.YELLOW),
    ' ': str(colorama.Style.DIM) + '*' + str(colorama.Style.RESET_ALL),
})
_COLOR_RESET = str(colorama.Style.RESET_ALL)


def list_to_spaced_str(items: Sequence[str]) -> str:
  """Returns a string with items joined and spaced every n characters."""
  spaced_list = []
  for i, item in enumerate(items):
    if _SPACE_EVERY_N.value > 0 and i % _SPACE_EVERY_N.value == 0 and i > 0:
      spaced_list.append(' ')
    spaced_list.append(item)
  return ''.join(spaced_list)


def colorful(seq: str) -> str:
  """Returns a colorful string of a sequence of DNA bases."""
  if not _COLOR.value:
    return list_to_spaced_str([base if base != ' ' else '*' for base in seq])
  return list_to_spaced_str([
      f'{_DNA_COLORS.get(base, "")}{base}{_COLOR_RESET}'
      if base != ' '
      else _DNA_COLORS[' ']
      for base in seq
  ])


def decode_bases(encoded_bases: np.ndarray) -> str:
  """Decodes encoded bases (floats) to string."""
  return ''.join([dc_constants.SEQ_VOCAB[int(b)] for b in encoded_bases])


def show_example(
    *,
    rows: tf.Tensor,
    label: tf.Tensor,
    num_passes: int,
    colorful_fn: Callable[[str], str] = colorful,
):
  """Shows a single example."""
  # rows shape: (height, width, 1)
  rows_np = np.squeeze(rows.numpy())
  height, width = rows_np.shape

  # Try to detect if ccs_bq is used.
  # total_rows = max_passes * 4 + (6 if use_ccs_bq else 5)
  if (height - 6) % 4 == 0:
    use_ccs_bq = True
    max_passes = (height - 6) // 4
  elif (height - 5) % 4 == 0:
    use_ccs_bq = False
    max_passes = (height - 5) // 4
  else:
    # Default fallback
    use_ccs_bq = False
    max_passes = (height - 5) // 4

  dc_config = pre_lib.DcConfig(max_passes, width, use_ccs_bq=use_ccs_bq)

  label_np = np.squeeze(label.numpy())
  if label_np.size > 0:
    label_str = decode_bases(label_np)
    print(f'Label\t{colorful_fn(label_str)}')
  else:
    label_str = None

  # CCS read
  ccs_idx = dc_config.indices('ccs')
  ccs_bases = rows_np[ccs_idx]
  if ccs_bases.ndim > 1:
    ccs_bases = ccs_bases[0]
  ccs_str = decode_bases(ccs_bases)
  print(f'CCS\t{colorful_fn(ccs_str)}')

  # Subreads
  bases_idx = dc_config.indices('bases', n_subreads=int(num_passes))
  subread_bases = rows_np[bases_idx]

  for i, subread_row in enumerate(subread_bases):
    if np.all(subread_row == 0):
      continue
    subread_str = decode_bases(subread_row)
    print(f'Read {i + 1}\t{colorful_fn(subread_str)}')


def visualize_examples(
    file_path: str,
    num_records_to_visualize: int,
    names: Sequence[str] | None = None,
) -> None:
  """Visualizes examples from tfrecord files."""

  # Get one example to determine max_length and height
  raw_dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
  for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    subreads_shape = example.features.feature['subreads/shape'].int64_list.value
    height, width, _ = subreads_shape
    # Check if it has labels
    has_label = 'label/encoded' in example.features.feature
    break
  else:
    return

  params = model_configs.get_config()
  with params.unlocked():
    params.max_length = width
    if (height - 6) % 4 == 0:
      params.use_ccs_bq = True
      params.max_passes = (height - 6) // 4
    else:
      params.use_ccs_bq = False
      params.max_passes = (height - 5) // 4
    params.total_rows = height
    params.buffer_size = 1
    params.remove_label_gaps = False

  dataset = data_providers.get_dataset(
      file_pattern=file_path,
      num_epochs=1,
      batch_size=1,
      params=params,
      inference=not has_label,
      limit=-1 if names else num_records_to_visualize,
      drop_remainder=False,
  )

  names_set = set(names or ())
  example_i = 0
  for data in dataset:
    if example_i >= num_records_to_visualize:
      break
    name = data['name'][0][0].numpy().decode('utf-8')
    if names_set and name not in names_set:
      continue

    label_np = np.squeeze(data['label'][0].numpy())
    internal_gaps_str = ''
    if label_np.size > 0:
      label_str = decode_bases(label_np)
      num_internal_gaps = label_str.rstrip(' ').count(' ')
      internal_gaps_str = f'; label internal gaps: {num_internal_gaps}'

    print(
        f'\n--- Example {example_i+1}: {name} '
        f'(num_passes: {data["num_passes"][0].numpy().item():.0f}'
        f'{internal_gaps_str}) ---'
    )
    show_example(
        rows=data['rows'][0],
        label=data['label'][0],
        num_passes=int(data['num_passes'][0].numpy().item()),
    )
    example_i += 1


def main(unused_argv: Sequence[str]) -> None:
  if len(unused_argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  visualize_examples(
      _FILE_PATH.value,
      _NUM_RECORDS.value,
      _NAMES.value,
  )


if __name__ == '__main__':
  app.run(main)
