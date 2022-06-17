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
"""Tests for deepconsensus.quality_calibration.calculate_baseq_calibration."""

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam
from deepconsensus.quality_calibration import calculate_baseq_calibration
from deepconsensus.utils import test_utils


class Test(parameterized.TestCase):
  """Test generate quality qq script."""

  @parameterized.parameters(
      dict(
          region_string="chr20:0-1000",
          expected_record=calculate_baseq_calibration.RegionRecord(
              "chr20", 0, 1000),
          message="Test 1: valid region string parsing"),
      dict(
          region_string="chr20",
          expected_record=calculate_baseq_calibration.RegionRecord(
              "chr20", 0, 200000),
          message="Test 2: valid region with contig only, no range provided."))
  @flagsaver.flagsaver
  def test_process_region_string(self, region_string, expected_record, message):
    """Test process region string method that parses a region string."""
    ref_file = "prediction_assessment/CHM13_chr20_0_200000.fa"
    fasta = test_utils.deepconsensus_testdata(ref_file)
    returned_record = calculate_baseq_calibration.process_region_string(
        region_string, fasta)

    self.assertEqual(
        expected_record.contig, returned_record.contig, msg=message)
    self.assertEqual(expected_record.start, returned_record.start, msg=message)
    self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  @parameterized.parameters(
      dict(
          region_string="chr20:1000-0",
          message="Test 3: Invalid contig region, stop is smaller than start."),
      dict(
          region_string="chr20:0-ABCD",
          message="Test 4: Invalid contig region, stop is not an int."),
      dict(
          region_string="chr20:0-1000#",
          message="Test 5: Invalid contig region, stop is not an int."),
      dict(region_string="chr20:0::-::10:0:0", message="Test 6: Invalid range"))
  @flagsaver.flagsaver
  def test_process_region_string_exeptions(self, region_string, message):
    """Test process region string method that parses a region string."""
    ref_file = "prediction_assessment/CHM13_chr20_0_200000.fa"
    fasta = test_utils.deepconsensus_testdata(ref_file)
    with self.assertRaises(Exception, msg=message):
      calculate_baseq_calibration.process_region_string(region_string, fasta)

  @parameterized.parameters(
      dict(
          input_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 1000),
              calculate_baseq_calibration.RegionRecord("chrX", 0, 1000)
          ],
          interval_length=500,
          expected_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 500),
              calculate_baseq_calibration.RegionRecord("chr20", 500, 1000),
              calculate_baseq_calibration.RegionRecord("chrX", 0, 500),
              calculate_baseq_calibration.RegionRecord("chrX", 500, 1000)
          ],
          message="Test1: Valid input with ranges that end without division"),
      dict(
          input_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 1100),
              calculate_baseq_calibration.RegionRecord("chrX", 0, 1098)
          ],
          interval_length=500,
          expected_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 500),
              calculate_baseq_calibration.RegionRecord("chr20", 500, 1000),
              calculate_baseq_calibration.RegionRecord("chr20", 1000, 1100),
              calculate_baseq_calibration.RegionRecord("chrX", 0, 500),
              calculate_baseq_calibration.RegionRecord("chrX", 500, 1000),
              calculate_baseq_calibration.RegionRecord("chrX", 1000, 1098)
          ],
          message="Test 2: End is not divible by interval length"),
      dict(
          input_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 358),
              calculate_baseq_calibration.RegionRecord("chrX", 0, 457)
          ],
          interval_length=1000,
          expected_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 358),
              calculate_baseq_calibration.RegionRecord("chrX", 0, 457)
          ],
          message="Test 3: Ranges where contig length is much smaller."))
  @flagsaver.flagsaver
  def test_split_regions_in_intervals(self, input_list, interval_length,
                                      expected_list, message):
    """Test split regions method that divides a region by a given interval length."""
    returned_list = calculate_baseq_calibration.split_regions_in_intervals(
        input_list, interval_length)

    for returned_record, expected_record in zip(returned_list, expected_list):
      self.assertEqual(
          expected_record.contig, returned_record.contig, msg=message)
      self.assertEqual(
          expected_record.start, returned_record.start, msg=message)
      self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  @parameterized.parameters(
      dict(
          region="chr20:0-1000",
          expected_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 0, 1000)
          ],
          message="Test 1: With a region."),
      dict(
          region="chr20:1324-2000",
          expected_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 1324, 2000)
          ],
          message="Test 2: A region that does not start at 0 and length is less than 1000"
      ),
      dict(
          region="chr20:67-123",
          expected_list=[
              calculate_baseq_calibration.RegionRecord("chr20", 67, 123)
          ],
          message="Test 3: A region that does not start at 0 and length is less than 1000"
      ))
  @flagsaver.flagsaver
  def test_get_contig_regions(self, region, expected_list, message):
    """Test get contig regions method that produces a list of regions for processing."""
    fasta = "prediction_assessment/CHM13_chr20_0_200000.fa"
    fasta_file = test_utils.deepconsensus_testdata(fasta)
    bam = "prediction_assessment/CHM13_chr20_0_200000_dc.to_truth.bam"
    bam_file = test_utils.deepconsensus_testdata(bam)
    interval_length = 1000
    returned_list = calculate_baseq_calibration.get_contig_regions(
        bam_file, fasta_file, region, interval_length)

    self.assertEqual(len(expected_list), len(returned_list))
    for returned_record, expected_record in zip(returned_list, expected_list):
      self.assertEqual(
          expected_record.contig, returned_record.contig, msg=message)
      self.assertEqual(
          expected_record.start, returned_record.start, msg=message)
      self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  def set_read_attributes(self, read):
    """Set attributes of reads that are redundant."""
    read.is_duplicate = False
    read.is_qcfail = False
    read.is_secondary = False
    read.is_unmapped = False
    read.is_supplementary = False
    read.mapping_quality = 60

  @parameterized.parameters(
      dict(
          query_sequence="AAAA",
          reference_name="chr20",
          region_interval=calculate_baseq_calibration.RegionRecord(
              "chr20", 0, 100),
          reference_start=0,
          query_qualities=[1, 2, 3, 4],
          cigartuples=[(pysam.CMATCH, 4)],
          ref_sequence="AAAA",
          max_qual=4,
          expected_dict=[{
              "M": 0,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }],
          message="Test 1: Simple test wil all match."),
      dict(
          query_sequence="AAAA",
          reference_name="chr20",
          region_interval=calculate_baseq_calibration.RegionRecord(
              "chr20", 1, 100),
          reference_start=0,
          query_qualities=[1, 2, 3, 4],
          cigartuples=[(pysam.CMATCH, 4)],
          ref_sequence="AAAA",
          max_qual=4,
          expected_dict=[{
              "M": 0,
              "X": 0
          }, {
              "M": 0,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }],
          message="Test 2: Start is one base off."),
      dict(
          query_sequence="AAAT",
          reference_name="chr20",
          region_interval=calculate_baseq_calibration.RegionRecord(
              "chr20", 1, 100),
          reference_start=0,
          query_qualities=[1, 2, 3, 4],
          cigartuples=[(pysam.CMATCH, 4)],
          ref_sequence="AAAA",
          max_qual=4,
          expected_dict=[{
              "M": 0,
              "X": 0
          }, {
              "M": 0,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 0,
              "X": 1
          }],
          message="Test 3: Last base is a mismatch"),
      dict(
          query_sequence="AACCAT",
          reference_name="chr20",
          region_interval=calculate_baseq_calibration.RegionRecord(
              "chr20", 0, 100),
          reference_start=0,
          query_qualities=[1, 2, 3, 3, 4, 5],
          cigartuples=[(pysam.CMATCH, 2), (pysam.CINS, 2), (pysam.CMATCH, 2)],
          ref_sequence="AAAA",
          max_qual=5,
          expected_dict=[{
              "M": 0,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 0,
              "X": 2
          }, {
              "M": 1,
              "X": 0
          }, {
              "M": 0,
              "X": 1
          }],
          message="Test 4: An insert, we consider inserts as mismatches"),
      dict(
          query_sequence="AACCAT",
          reference_name="chr20",
          region_interval=calculate_baseq_calibration.RegionRecord(
              "chr20", 0, 100),
          reference_start=0,
          query_qualities=[1, 2, 3, 3, 4, 5],
          cigartuples=[(pysam.CMATCH, 2), (pysam.CINS, 2), (pysam.CMATCH, 2)],
          ref_sequence="GGGG",
          max_qual=5,
          expected_dict=[{
              "M": 0,
              "X": 0
          }, {
              "M": 0,
              "X": 1
          }, {
              "M": 0,
              "X": 1
          }, {
              "M": 0,
              "X": 2
          }, {
              "M": 0,
              "X": 1
          }, {
              "M": 0,
              "X": 1
          }],
          message="Test 5: Everything is a mismatch."))
  @flagsaver.flagsaver
  def test_get_quality_calibration_stats(self, query_sequence, reference_name,
                                         reference_start, query_qualities,
                                         cigartuples, ref_sequence,
                                         region_interval, max_qual,
                                         expected_dict, message):
    """Test the base quality calibration calculation method."""
    bam = "prediction_assessment/CHM13_chr20_0_200000_dc.to_truth.bam"
    bam_file = test_utils.deepconsensus_testdata(bam)
    pysam_bam = pysam.AlignmentFile(bam_file)
    min_mapq = 60
    # test 1: Simple test for 4 baseq where all bases match
    read_1 = pysam.AlignedSegment(pysam_bam.header)
    self.set_read_attributes(read_1)
    read_1.query_sequence = query_sequence
    read_1.reference_name = reference_name
    read_1.reference_start = reference_start
    read_1.query_qualities = query_qualities
    read_1.cigartuples = cigartuples
    reads = [read_1]
    match_mismatch_dict = calculate_baseq_calibration.get_quality_calibration_stats(
        reads, ref_sequence, region_interval, min_mapq)

    for baseq in range(0, max_qual + 1):
      for match_mismatch in ["M", "X"]:
        self.assertEqual(
            match_mismatch_dict[baseq][match_mismatch],
            expected_dict[baseq][match_mismatch],
            msg=message)

  @parameterized.parameters(
      dict(
          is_duplicate=True,
          mapping_quality=60,
          message="Test 1: Duplicate read"),
      dict(
          is_duplicate=False,
          mapping_quality=59,
          message="Test 2: Read low mapq"))
  @flagsaver.flagsaver
  def test_get_quality_calibration_stats_invalid_reads(self, is_duplicate,
                                                       mapping_quality,
                                                       message):
    """Test the base quality calibration calculation method."""
    bam = "prediction_assessment/CHM13_chr20_0_200000_dc.to_truth.bam"
    bam_file = test_utils.deepconsensus_testdata(bam)
    pysam_bam = pysam.AlignmentFile(bam_file)
    min_mapq = 60

    read_1 = pysam.AlignedSegment(pysam_bam.header)
    self.set_read_attributes(read_1)
    read_1.query_sequence = "AAAA"
    read_1.reference_name = "chr20"
    read_1.reference_start = 0
    read_1.query_qualities = [1, 2, 3, 4]
    read_1.cigartuples = [(pysam.CMATCH, 4)]
    read_1.is_duplicate = is_duplicate
    read_1.mapping_quality = mapping_quality
    reads = [read_1]
    ref_sequence = "AAAA"
    min_mapq = 60
    max_qual = 4
    region_interval = calculate_baseq_calibration.RegionRecord("chr20", 0, 100)
    match_mismatch_dict = calculate_baseq_calibration.get_quality_calibration_stats(
        reads, ref_sequence, region_interval, min_mapq)

    # everything should be zero as this read will be skipped
    for baseq in range(0, max_qual + 1):
      for match_mismatch in ["M", "X"]:
        self.assertEqual(
            match_mismatch_dict[baseq][match_mismatch], 0, msg=message)


if __name__ == "__main__":
  absltest.main()
