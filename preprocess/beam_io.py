# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Beam sources for genomics file formats."""

from apache_beam import metrics
from apache_beam import transforms
from apache_beam.io import filebasedsource
from apache_beam.io.iobase import Read
import tensorflow as tf

from nucleus.io import bed
from nucleus.io import fasta
from nucleus.io import sam


class _GenomicsSource(filebasedsource.FileBasedSource):
  """A base source for reading genomics files.

  Do not use this class directly. Instead, use the subclass for the specific
  file type.
  """

  def __init__(self, file_pattern, validate, **nucleus_kwargs):
    """Initialize a _GenomicsSource for use with readers for genomics files."""

    super(_GenomicsSource, self).__init__(
        file_pattern=file_pattern, splittable=False, validate=validate)
    self.nucleus_kwargs = nucleus_kwargs

  def read_records(self, input_path, offset_range_tracker):
    """Yield records returned by nucleus_reader."""
    if offset_range_tracker.start_position():
      raise ValueError('Start position not 0: %d' %
                       offset_range_tracker.start_position())
    current_offset = offset_range_tracker.start_position()
    reader = self.nucleus_reader(input_path, **self.nucleus_kwargs)

    with reader:
      for record in reader:
        if not offset_range_tracker.try_claim(current_offset):
          raise RuntimeError('Unable to claim position: %d' % current_offset)
        yield record
        current_offset += 1

  @property
  def nucleus_reader(self):
    raise NotImplementedError


class _SamSource(_GenomicsSource):
  """A source for reading SAM/BAM files."""

  nucleus_reader = sam.SamReader


class _IndexedFastaSource(_GenomicsSource):
  """A source for reading FASTA files containing a reference genome."""

  nucleus_reader = fasta.IndexedFastaReader


class _BedSource(_GenomicsSource):
  """A source for reading FASTA files containing a reference genome."""

  nucleus_reader = bed.BedReader


class ReadGenomicsFile(transforms.PTransform):
  """For reading one or more genomics files.

  Do not use this class directly. Instead, use the subclass for the specific
  file type.
  """

  def __init__(self, file_pattern, validate=True, **nucleus_kwargs):
    """Initialize the ReadSam transform."""

    super(ReadGenomicsFile, self).__init__()
    self._source = self._source_class(
        file_pattern, validate=validate, **nucleus_kwargs)

  def expand(self, pvalue):
    return pvalue.pipeline | Read(self._source)

  @property
  def _source_class(self):
    raise NotImplementedError


class ReadSam(ReadGenomicsFile):
  """For reading reads from one or more SAM/BAM files."""

  _source_class = _SamSource


class PlainTextFastaSource(filebasedsource.FileBasedSource):
  """Reads a plaintext fasta file (no compression; parallel readers)."""

  def __init__(self, file_pattern):
    """Initialize a plain-text FASTA reader."""
    super(PlainTextFastaSource, self).__init__(
        file_pattern=file_pattern, splittable=False, validate=True)
    self.fasta_records_counter = metrics.Metrics.counter(
        self.__class__, 'label_fasta_records')

  def read_records(self, input_path, offset_range_tracker):
    """Yield fasta records from a plaintext file."""
    start_offset = offset_range_tracker.start_position()
    with tf.io.gfile.GFile(input_path, 'r') as f:
      f.seek(start_offset)
      # Read the first key
      line = f.readline()
      key = line.strip('>\n')
      seq = ''
      while True:
        line = f.readline()
        if line.startswith('>'):
          # New record reached.
          yield (key, seq)
          self.fasta_records_counter.inc()
          key = line.strip('>\n')
          seq = ''
        else:
          seq += line.strip()

        if not line:
          # End of file reached; Yield and break from loop
          yield (key, seq)
          self.fasta_records_counter.inc()
          break


class ReadFastaFile(transforms.PTransform):
  """PTransform for parallel plaintext fasta reading."""

  def __init__(self, file_pattern):
    super(ReadFastaFile, self).__init__()
    self.file_pattern = file_pattern

  def expand(self, pcoll):
    return pcoll.pipeline | Read(PlainTextFastaSource(self.file_pattern))


class ReadIndexedFasta(ReadGenomicsFile):
  """For reading sequences from one or more IndexedFasta files."""

  _source_class = _IndexedFastaSource


class ReadBed(ReadGenomicsFile):
  """For reading sequences from one or more Bed files."""

  _source_class = _BedSource
