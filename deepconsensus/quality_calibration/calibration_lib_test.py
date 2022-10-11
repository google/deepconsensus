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
"""Tests for calibration_lib."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from deepconsensus.quality_calibration import calibration_lib


class CalibrateLibTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          calibration_str='skip',
          expected=calibration_lib.QualityCalibrationValues(
              enabled=False, threshold=0.0, w=1.0, b=0.0),
          message='Test 1: Valid empty calibration string.'),
      dict(
          calibration_str='10,1.0,0.2222',
          expected=calibration_lib.QualityCalibrationValues(
              enabled=True, threshold=10.0, w=1.0, b=0.2222),
          message='Test 2: Valid calibration string with positive values.'),
      dict(
          calibration_str='-10,1.0,0.2222',
          expected=calibration_lib.QualityCalibrationValues(
              enabled=True, threshold=-10.0, w=1.0, b=0.2222),
          message='Test 3: Valid calibration string with negative threshold.'),
      dict(
          calibration_str='-10,-1.0,-0.2222',
          expected=calibration_lib.QualityCalibrationValues(
              enabled=True, threshold=-10.0, w=-1.0, b=-0.2222),
          message='Test 4: Valid calibration string with all negative values.'))
  def test_parse_calibration_string(self, calibration_str, expected, message):
    """Tests for parse_calibration_string method."""
    returned = calibration_lib.parse_calibration_string(calibration_str)
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
          message='Test 3: Invalid calibration string 10,1.0.'),
      dict(
          calibration_str='10,AB,1.0',
          message='Test 4: Invalid calibration string 10,AB,1.0.'),
      dict(
          calibration_str='10,0.1.1,1.0',
          message='Test 5: Invalid calibration string 10,0.1.1,1.0.'),
  )
  def test_parse_calibration_string_exceptions(self, calibration_str, message):
    with self.assertRaises(Exception, msg=message):
      calibration_lib.parse_calibration_string(calibration_str)

  @parameterized.parameters(
      dict(
          input_values=np.array([0, 1, 2, 3, 4]),
          calibration_str='0,0,1',
          expected_output=np.array([1, 1, 1, 1, 1])),
      dict(
          input_values=np.array([0, 1, 2, 3, 4]),
          calibration_str='0,1,1',
          expected_output=np.array([1, 2, 3, 4, 5])),
      dict(
          input_values=np.array([0, 1, 2, 3, 4, 5]),
          calibration_str='3,1,1',
          expected_output=np.array([0, 1, 2, 3, 5, 6])),
  )
  def test_calibration(self, input_values, calibration_str, expected_output):
    calibration_values = calibration_lib.parse_calibration_string(
        calibration_str)
    output = calibration_lib.calibrate_quality_scores(input_values,
                                                      calibration_values)
    self.assertTrue(np.array_equal(output, expected_output))


if __name__ == '__main__':
  absltest.main()
