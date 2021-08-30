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
"""Tests for deepconsensus.preprocess.generate_input_transforms."""


from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as beam_testing_util

from deepconsensus.preprocess import generate_input_transforms
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils as dc_test_utils
from nucleus.protos import bed_pb2
from nucleus.testing import test_utils


class GetReadMoleculeNameDoFnTest(absltest.TestCase):

  def test_get_read_molecule_name(self):
    """Tests that molecule name for each read comes from the fragment name."""

    with test_pipeline.TestPipeline() as p:
      reads = [
          test_utils.make_read(
              bases='ATCG',
              start=0,
              chrom='m54316_180808_005743/1/ccs',
              name='m54316_180808_005743/2/truth'),
          test_utils.make_read(
              bases='ATCG',
              start=0,
              chrom='m54316_180808_005743/3/ccs',
              name='m54316_180808_005743/4/truth'),
      ]
      read_names = (
          p
          | beam.Create(reads)
          | beam.ParDo(generate_input_transforms.GetReadMoleculeNameDoFn()))

      expected = [
          ('m54316_180808_005743/2', reads[0]),
          ('m54316_180808_005743/4', reads[1]),
      ]
      beam_testing_util.assert_that(read_names,
                                    beam_testing_util.equal_to(expected))


class ExpandFieldsRemoveSoftClipsDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='label with all M',
          subread=test_utils.make_read(
              name='read 1', bases='ATCG', start=0, cigar='4M'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM',
              pw=None,
              ip=None),
          is_label=True,
      ),
      dict(
          testcase_name='subread with all M and no PW/IP present',
          subread=test_utils.make_read(
              name='read 1', bases='ATCG', start=0, cigar='4M'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM',
              pw=[],
              ip=[]),
          is_label=False,
      ),
      dict(
          testcase_name='subread with all M and PW/IP present',
          subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='ATCG',
              start=0,
              cigar='4M',
              pw=[1, 2, 3, 4],
              ip=[5, 6, 7, 8]),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM',
              pw=[1, 2, 3, 4],
              ip=[5, 6, 7, 8]),
          is_label=False,
      ),
      dict(
          testcase_name='label with complex cigar',
          subread=test_utils.make_read(
              name='read 5', bases='ATCGTT', start=0, cigar='2S1M1D2M1D1I'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 5',
              bases='C%sGT%sT' %
              (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD),
              start=0,
              cigar='1M1D2M1D1I',
              expanded_cigar='MDMMDI',
              pw=None,
              ip=None),
          is_label=True,
      ),
      dict(
          testcase_name='read with complex cigar and PW/IP',
          subread=dc_test_utils.make_read_with_info(
              name='read 5',
              bases='ATCGTT',
              start=0,
              cigar='2S1M1D2M1D1I',
              # PW/IP values only present for non softclipped bases.
              pw=[1, 2, 3, 4, 5, 6],
              ip=[7, 8, 9, 10, 11, 12]),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 5',
              bases='C%sGT%sT' %
              (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD),
              start=0,
              cigar='1M1D2M1D1I',
              expanded_cigar='MDMMDI',
              pw=[3, 4, 5, 6],
              ip=[9, 10, 11, 12]),
          is_label=False,
      ))
  def test_expand_fields_remove_soft_clips(self, subread, expected_subread,
                                           is_label):
    """Tests that sequence and cigar expanded and soft clips removed."""

    molecule_name = 'molecule name'
    molecule_data = (molecule_name, [subread])
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(
              generate_input_transforms.ExpandFieldsRemoveSoftClipsDoFn(
                  is_label=is_label)))
      expected = [(molecule_name, [expected_subread])]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class IndentReadsDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='start is 0',
          subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM')),
      dict(
          testcase_name='start > 0',
          subread=dc_test_utils.make_read_with_info(
              name='read 2',
              bases='GTTA',
              start=3,
              cigar='2M2I',
              expanded_cigar='MMII'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 2',
              bases='%sGTTA' % (dc_constants.GAP_OR_PAD * 3),
              start=3,
              cigar='2M2I',
              expanded_cigar='%sMMII' % (dc_constants.GAP_OR_PAD * 3))),
  )
  def test_indentation(self, subread, expected_subread):
    """Tests that subreads are indented correctly based on start position."""

    molecule_name = 'molecule name'
    molecule_data = (molecule_name, [subread])
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(generate_input_transforms.IndentReadsDoFn()))
      expected = [(molecule_name, [expected_subread])]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class AlignSubreadSequencesDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='two subreads with same sequence',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M')
          ] * 2,
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M',
              )
          ] * 2),
      dict(
          testcase_name='two subreads with different lengths',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTA',
                  start=0,
                  cigar='4M'),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M'),
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTA',
                  start=0,
                  cigar='4M',
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M',
              ),
          ]),
      dict(
          testcase_name='two subreads with one I',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M',
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMIM',
                  bases='ACTAG',
                  start=0,
                  cigar='3M1I1M'),
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar=f'MMM{dc_constants.GAP_OR_PAD}M',
                  bases=f'ACT{dc_constants.GAP_OR_PAD}G',
                  start=0,
                  cigar='4M',
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMIM',
                  bases='ACTAG',
                  start=0,
                  cigar='3M1I1M'),
          ]),
      dict(
          testcase_name='two subreads with one D',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M'),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMDM',
                  bases='ACT%sG' % dc_constants.GAP_OR_PAD,
                  start=0,
                  cigar='3M1D1M'),
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M',
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMDM',
                  bases='ACT%sG' % dc_constants.GAP_OR_PAD,
                  start=0,
                  cigar='3M1D1M')
          ]),
      dict(
          testcase_name='two subreads with D and I',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='DMMM',
                  bases='%sCTG' % dc_constants.GAP_OR_PAD,
                  start=0,
                  cigar='1D3M',
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='IMMMM',
                  bases='GACTG',
                  start=0,
                  cigar='1I4M')
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar=' DMMM',
                  bases=f'{dc_constants.GAP_OR_PAD * 2}CTG',
                  start=0,
                  cigar='1D3M',
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='IMMMM',
                  bases='GACTG',
                  start=0,
                  cigar='1I4M'),
          ]))
  def test_align_subread_sequences_do_fn_test(
      self,
      subreads,
      expected_subreads,
  ):
    """Tests that sequences are aligned correctly."""

    molecule_name = 'some molecule'
    molecule_data = (molecule_name, subreads)
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(generate_input_transforms.AlignSubreadSequencesDoFn()))
      expected = [(molecule_name, expected_subreads)]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class AlignLabelSequencesDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='subread and label with same sequence',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M')
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M',
                  subread_indices=[1, 2, 3, 4],
              )
          ],
          label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMM',
              bases='ACTG',
              start=0,
              cigar='4M'),
          expected_label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMM',
              bases='ACTG',
              start=0,
              cigar='4M')),
      dict(
          testcase_name='subread and label with different lengths',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTA',
                  start=0,
                  cigar='4M'),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M'),
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTA',
                  start=0,
                  cigar='4M',
                  subread_indices=[1, 2, 3, 4, 5],
              ),
              dc_test_utils.make_read_with_info(
                  name='read 2',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M',
                  subread_indices=[1, 2, 3, 4, 5],
              ),
          ],
          label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMM',
              bases='ACTG',
              start=0,
              cigar='4M'),
          expected_label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMM',
              bases='ACTG',
              start=0,
              cigar='4M')),
      dict(
          testcase_name='subread and label with one I',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M',
                  unsup_insertions_by_pos_keys=[2],
                  unsup_insertions_by_pos_values=[1],
              )
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMM',
                  bases='ACTG',
                  start=0,
                  cigar='4M',
                  subread_indices=[1, 2, 4, 5],
                  unsup_insertions_by_pos_keys=[2],
                  unsup_insertions_by_pos_values=[1],
              ),
          ],
          label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMIM',
              bases='ACTAG',
              start=0,
              cigar='3M1I1M'),
          expected_label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMIM',
              bases='ACTAG',
              start=0,
              cigar='3M1I1M')),
      dict(
          testcase_name='subread and label with one D',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M')
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='MMMMM',
                  bases='ACTAG',
                  start=0,
                  cigar='5M',
                  subread_indices=[1, 2, 3, 4, 5],
              )
          ],
          label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMDM',
              bases='ACT%sG' % dc_constants.GAP_OR_PAD,
              start=0,
              cigar='3M1D1M'),
          expected_label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='MMMDM',
              bases='ACT%sG' % dc_constants.GAP_OR_PAD,
              start=0,
              cigar='3M1D1M')),
      dict(
          testcase_name='subread and label with D and I',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='DMMM',
                  bases='%sCTG' % dc_constants.GAP_OR_PAD,
                  start=0,
                  cigar='1D3M',
                  unsup_insertions_by_pos_keys=[0],
                  unsup_insertions_by_pos_values=[1],
              )
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read 1',
                  expanded_cigar='DMMM',
                  bases='%sCTG' % dc_constants.GAP_OR_PAD,
                  start=0,
                  cigar='1D3M',
                  subread_indices=[2, 3, 4, 5],
                  unsup_insertions_by_pos_keys=[0],
                  unsup_insertions_by_pos_values=[1],
              )
          ],
          label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='IMMMM',
              bases='GACTG',
              start=0,
              cigar='1I4M'),
          expected_label=dc_test_utils.make_read_with_info(
              name='label',
              expanded_cigar='IMMMM',
              bases='GACTG',
              start=0,
              cigar='1I4M')),
  )
  def test_align_label_sequences_do_fn_test(self, subreads, expected_subreads,
                                            label, expected_label):
    """Tests that sequences are aligned correctly."""

    molecule_name = 'some molecule'
    molecule_data = (molecule_name, ([subreads], [[label]]))
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(generate_input_transforms.AlignLabelSequencesDoFn()))
      expected = [(molecule_name, (expected_subreads, [expected_label]))]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class PadSubreadsDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='same read lengths',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read',
                  bases='ATCG',
                  start=0,
                  cigar='4M',
                  expanded_cigar='MMMM')
          ] * 2,
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read',
                  bases='ATCG',
                  start=0,
                  cigar='4M',
                  expanded_cigar='MMMM')
          ] * 2,
      ),
      dict(
          testcase_name='different read lengths',
          subreads=[
              dc_test_utils.make_read_with_info(
                  name='read',
                  bases='ATCG',
                  start=0,
                  cigar='4M',
                  expanded_cigar='MMMM'),
              dc_test_utils.make_read_with_info(
                  name='read',
                  bases='ATCG%s' % (dc_constants.GAP_OR_PAD * 2),
                  start=0,
                  cigar='4M',
                  expanded_cigar='MMMM%s' % (dc_constants.GAP_OR_PAD * 2))
          ],
          expected_subreads=[
              dc_test_utils.make_read_with_info(
                  name='read',
                  bases='ATCG%s' % (dc_constants.GAP_OR_PAD * 2),
                  start=0,
                  cigar='4M',
                  expanded_cigar='MMMM%s' % (dc_constants.GAP_OR_PAD * 2))
          ] * 2,
      ),
  )
  def test_padding(self, subreads, expected_subreads):
    """Tests that reads correctly padded at ends to be of same length."""

    molecule_name = 'molecule name'
    molecule_data = (molecule_name, subreads)
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(generate_input_transforms.PadSubreadsDoFn()))

      expected = [(molecule_name, expected_subreads)]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class PadSubreadsAndLabelDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='same read lengths',
          subread=dc_test_utils.make_read_with_info(
              name='read 3',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM',
              pw=[1, 2, 3, 4],
              ip=[1, 2, 3, 4]),
          label=dc_test_utils.make_read_with_info(
              name='label 3',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 3',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM',
              pw=[1, 2, 3, 4],
              ip=[1, 2, 3, 4]),
          expected_label=dc_test_utils.make_read_with_info(
              name='label 3',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM'),
      ),
      dict(
          testcase_name='different read lengths',
          subread=dc_test_utils.make_read_with_info(
              name='read 4',
              bases='ATCG',
              start=0,
              cigar='4M',
              expanded_cigar='MMMM',
              pw=[1, 2, 3, 4],
              ip=[1, 2, 3, 4]),
          label=dc_test_utils.make_read_with_info(
              name='label 4',
              bases='ATCGTT',
              start=0,
              cigar='4M2I',
              expanded_cigar='MMMMII'),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 4',
              bases='ATCG%s' % (dc_constants.GAP_OR_PAD * 2),
              start=0,
              cigar='4M',
              expanded_cigar='MMMM%s' % (dc_constants.GAP_OR_PAD * 2),
              pw=[1, 2, 3, 4] + [dc_constants.GAP_OR_PAD_INT] * 2,
              ip=[1, 2, 3, 4] + [dc_constants.GAP_OR_PAD_INT] * 2),
          expected_label=dc_test_utils.make_read_with_info(
              name='label 4',
              bases='ATCGTT',
              start=0,
              cigar='4M2I',
              expanded_cigar='MMMMII',
          )),
  )
  def test_padding(self, subread, label, expected_subread, expected_label):
    """Tests that reads correctly padded at ends to be of same length."""

    molecule_name = 'molecule name'
    molecule_data = (molecule_name, ([subread], [label]))
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(generate_input_transforms.PadSubreadsAndLabelDoFn()))

      expected = [(molecule_name, ([expected_subread], [expected_label]))]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class AlignPwIpDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='read with external pad',
          subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='%sATCG' % dc_constants.GAP_OR_PAD,
              start=1,
              cigar='4M',
              expanded_cigar='%sMMMM' % dc_constants.GAP_OR_PAD,
              pw=[1, 2, 4, 1],
              ip=[6, 5, 8, 9]),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='%sATCG' % dc_constants.GAP_OR_PAD,
              start=1,
              cigar='4M',
              expanded_cigar='%sMMMM' % dc_constants.GAP_OR_PAD,
              pw=[dc_constants.GAP_OR_PAD_INT, 1, 2, 4, 1],
              ip=[dc_constants.GAP_OR_PAD_INT, 6, 5, 8, 9]),
      ),
      dict(
          testcase_name='read with internal gap',
          subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='A%sTCG' % dc_constants.GAP_OR_PAD,
              start=0,
              cigar='1M1D3M',
              expanded_cigar='M%sMMM' % dc_constants.GAP_OR_PAD,
              pw=[1, 2, 4, 1],
              ip=[6, 5, 8, 9]),
          expected_subread=dc_test_utils.make_read_with_info(
              name='read 1',
              bases='A%sTCG' % dc_constants.GAP_OR_PAD,
              start=0,
              cigar='1M1D3M',
              expanded_cigar='M%sMMM' % dc_constants.GAP_OR_PAD,
              pw=[1, dc_constants.GAP_OR_PAD_INT, 2, 4, 1],
              ip=[6, dc_constants.GAP_OR_PAD_INT, 5, 8, 9]),
      ),
  )
  def test_align_pw_and_ip(self, subread, expected_subread):
    """Tests that pw and ip correctly aligned."""

    molecule_name = 'molecule name'
    molecule_data = (molecule_name, [subread])
    with test_pipeline.TestPipeline() as p:
      pipeline_output = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(generate_input_transforms.AlignPwIpDoFn()))

      # Label does not have pw or ip fields, so we expect it to be unchanged.
      expected = [(molecule_name, [expected_subread])]
      beam_testing_util.assert_that(pipeline_output,
                                    beam_testing_util.equal_to(expected))


class GetBedRecordMoleculeNameDoFnTest(absltest.TestCase):

  def test_get_bed_record_molecule_name(self):
    """Tests that molecule name correctly extracted from BedRecord proto."""

    with test_pipeline.TestPipeline() as p:
      bed_record = bed_pb2.BedRecord(
          reference_name='ecoliK12_pbi_August2018',
          start=0,
          end=10,
          name='m54316_180808_005743/7012875/ccs')
      molecule_name_and_record = (
          p
          | beam.Create([bed_record])
          | beam.ParDo(
              generate_input_transforms.GetBedRecordMoleculeNameDoFn()))

      expected = [('m54316_180808_005743/7012875', bed_record)]
      beam_testing_util.assert_that(molecule_name_and_record,
                                    beam_testing_util.equal_to(expected))


class CreateTrainDeepConsensusInputDoFnTest(absltest.TestCase):

  def test_create_deepconsensus_input(self):
    """Tests deepconsensus_pb2.DeepConsensusInput protos created correctly."""

    with test_pipeline.TestPipeline() as p:
      molecule_name = 'm54316_180808_005743/7012875'
      subread_indices = [1, 2, 3, 4]
      bed_record = bed_pb2.BedRecord(
          reference_name='ecoliK12_pbi_August2018',
          start=0,
          end=3,
          name='m54316_180808_005743/7012875/ccs',
          strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND)
      subreads = dc_test_utils.make_read_with_info(
          name='read 1',
          start=0,
          bases='ATCG',
          cigar='4M',
          expanded_cigar='MMMM',
          subread_strand=deepconsensus_pb2.Subread.REVERSE,
          ip=[1, 2, 3, 4],
          pw=[5, 6, 7, 8],
          sn=[0.1, 0.2, 0.3, 0.4],
          subread_indices=subread_indices)
      label = dc_test_utils.make_read_with_info(
          name='label',
          start=0,
          bases='ATCG',
          cigar='4M',
          expanded_cigar='MMMM',
          subread_strand=deepconsensus_pb2.Subread.FORWARD,
          ip=[],
          pw=[],
          sn=[])

      molecule_data = (molecule_name, ([([subreads], [label])], [bed_record]))
      deepconsensus_input = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(
              generate_input_transforms.CreateTrainDeepConsensusInputDoFn()))

      expected = [
          dc_test_utils.make_deepconsensus_input(
              subread_bases=['ATCG'],
              subread_expanded_cigars=['MMMM'],
              ips=[[1, 2, 3, 4]],
              pws=[[5, 6, 7, 8]],
              label_bases='ATCG',
              label_expanded_cigar='MMMM',
              molecule_name=molecule_name,
              molecule_start=0,
              subread_strand=[deepconsensus_pb2.Subread.REVERSE],
              chrom_name='ecoliK12_pbi_August2018',
              chrom_start=0,
              chrom_end=3,
              sn=[0.1, 0.2, 0.3, 0.4],
              strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND,
              subread_indices=subread_indices,
          )
      ]
      beam_testing_util.assert_that(deepconsensus_input,
                                    beam_testing_util.equal_to(expected))


class CreateInferenceDeepConsensusInputDoFnTest(absltest.TestCase):

  def test_create_deepconsensus_input(self):
    """Tests deepconsensus_pb2.DeepConsensusInput protos created correctly."""

    with test_pipeline.TestPipeline() as p:
      molecule_name = 'm54316_180808_005743/7012875'
      subreads = dc_test_utils.make_read_with_info(
          name='read 1',
          start=0,
          bases='ATCG',
          cigar='4M',
          expanded_cigar='MMMM',
          subread_strand=deepconsensus_pb2.Subread.REVERSE,
          ip=[1, 2, 3, 4],
          pw=[5, 6, 7, 8],
          sn=[0.1, 0.2, 0.3, 0.4])

      molecule_data = (molecule_name, [subreads])
      deepconsensus_input = (
          p
          | beam.Create([molecule_data])
          | beam.ParDo(
              generate_input_transforms.CreateInferenceDeepConsensusInputDoFn())
      )

      # Values used only for training will be empty, so fill in the default
      # values for that field type.
      dc_input = dc_test_utils.make_deepconsensus_input(
          inference=True,
          subread_bases=['ATCG'],
          subread_expanded_cigars=['MMMM'],
          ips=[[1, 2, 3, 4]],
          pws=[[5, 6, 7, 8]],
          molecule_name=molecule_name,
          molecule_start=0,
          subread_strand=[deepconsensus_pb2.Subread.REVERSE],
          sn=[0.1, 0.2, 0.3, 0.4],
          subread_indices=[])
      beam_testing_util.assert_that(deepconsensus_input,
                                    beam_testing_util.equal_to([dc_input]))


class AddCcsSequenceDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Simple.',
          dc_input=dc_test_utils.make_deepconsensus_input(
              subread_bases=['ATCGA'], subread_expanded_cigars=['MMMMM']),
          ccs_sequence='ATCGA',
          expected_ccs_sequence='ATCGA',
      ),
      dict(
          testcase_name='Subreads contain insertion.',
          dc_input=dc_test_utils.make_deepconsensus_input(
              subread_bases=['ATCGA'], subread_expanded_cigars=['MMIMM']),
          ccs_sequence='ATGA',
          expected_ccs_sequence='AT GA',
      ),
      dict(
          testcase_name='Add padding to CCS.',
          dc_input=dc_test_utils.make_deepconsensus_input(
              subread_bases=['ATCG '], subread_expanded_cigars=['MMMM ']),
          ccs_sequence='ATCG',
          expected_ccs_sequence='ATCG ',
      ),
  )
  def test_add_ccs_sequence(self, dc_input, ccs_sequence,
                            expected_ccs_sequence):
    """Checks that CCS sequence is correcrly added to the dc_input."""
    with test_pipeline.TestPipeline() as p:
      inputs = ('molecule_name', ([dc_input], [ccs_sequence]))
      outputs = (
          p
          | beam.Create([inputs])
          | beam.ParDo(generate_input_transforms.AddCcsSequenceDoFn()))
      dc_input.ccs_sequence = expected_ccs_sequence
      beam_testing_util.assert_that(outputs,
                                    beam_testing_util.equal_to([dc_input]))


if __name__ == '__main__':
  absltest.main()
