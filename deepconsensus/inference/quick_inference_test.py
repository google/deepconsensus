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
"""Tests for quick_inference."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam

from deepconsensus.inference import quick_inference
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils

FLAGS = flags.FLAGS


def setUpModule():
  logging.set_verbosity(logging.FATAL)


class QuickInferenceTest(parameterized.TestCase):


  @parameterized.parameters(
      dict(
          calibration_str='',
          expected=quick_inference.QualityCalibrationValues(
              enabled=False, threshold=0.0, w=1.0, b=0.0),
          message='Test 1: Valid empty calibration string.'),
      dict(
          calibration_str='10,1.0,0.2222',
          expected=quick_inference.QualityCalibrationValues(
              enabled=True, threshold=10.0, w=1.0, b=0.2222),
          message='Test 2: Valid calibration string with positive values.'),
      dict(
          calibration_str='-10,1.0,0.2222',
          expected=quick_inference.QualityCalibrationValues(
              enabled=True, threshold=-10.0, w=1.0, b=0.2222),
          message='Test 3: Valid calibration string with negative threshold.'),
      dict(
          calibration_str='-10,-1.0,-0.2222',
          expected=quick_inference.QualityCalibrationValues(
              enabled=True, threshold=-10.0, w=-1.0, b=-0.2222),
          message='Test 4: Valid calibration string with all negative values.'))
  @flagsaver.flagsaver
  def test_parse_calibration_string(self, calibration_str, expected, message):
    """Tests for parse_calibration_string method."""
    returned = quick_inference.parse_calibration_string(calibration_str)
    self.assertEqual(returned.enabled, expected.enabled, msg=message)
    self.assertEqual(returned.threshold, expected.threshold, msg=message)
    self.assertEqual(returned.w, expected.w, msg=message)
    self.assertEqual(returned.b, expected.b, msg=message)

  @parameterized.parameters(
      dict(
          calibration_str='ABCD',
          message='Test 1: Invalid calibration string ABCD.'),
      dict(
          calibration_str='A,BC,D',
          message='Test 2: Invalid calibration string A,BC,D.'),
      dict(
          calibration_str='10,1.0',
          message='Test 2: Invalid calibration string 10,1.0.'),
      dict(
          calibration_str='10,AB,1.0',
          message='Test 2: Invalid calibration string 10,AB,1.0.'),
      dict(
          calibration_str='10,0.1.1,1.0',
          message='Test 2: Invalid calibration string 10,0.1.1,1.0.'),
  )
  @flagsaver.flagsaver
  def test_parse_calibration_string_exceptions(self, calibration_str, message):
    with self.assertRaises(Exception, msg=message):
      quick_inference.parse_calibration_string(calibration_str)


if __name__ == '__main__':
  absltest.main()
