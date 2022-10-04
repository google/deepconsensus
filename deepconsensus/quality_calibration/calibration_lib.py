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
"""Library containing functions for calibrating base quality scores."""

import dataclasses

import numpy as np


@dataclasses.dataclass
class QualityCalibrationValues:
  """A structure that defines variables required for base quality calibration.

  Attributes:
    enabled: If set then calibration is enabled.
    threshold: A threshold value above which the qualities will be calibrated
    w: Coefficient for linear transformation.
    b: Bias term for linear transformation.
  """
  enabled: bool
  threshold: float
  w: float
  b: float


def parse_calibration_string(calibration: str) -> QualityCalibrationValues:
  """Parses calibration string and returns the threshold, w and b values."""
  # If calibration string is set to skip, no calibration will be performed.
  if calibration == 'skip':
    return QualityCalibrationValues(enabled=False, threshold=0.0, w=1.0, b=0.0)

  parsed_list = calibration.split(',')
  if len(parsed_list) != 3:
    raise ValueError(
        'Malformed calibration string. Expected 3 values (or set '
        'to "skip" to perform no quality calibration).', calibration)

  calibration_values = QualityCalibrationValues(
      enabled=True,
      threshold=float(parsed_list[0]),
      w=float(parsed_list[1]),
      b=float(parsed_list[2]))
  return calibration_values


def calibrate_quality_scores(
    quality_scores: np.ndarray,
    calibration_values: QualityCalibrationValues) -> np.ndarray:
  """Calibrate the quality score using linear transformation.

  Args:
    quality_scores: A list containing the predicted quality score.
    calibration_values: Coefficient values of the linear transformation.

  Returns:
    A list of calibrated quality scores.
  """
  if calibration_values.threshold == 0:
    # Skip O(n) operations of np.where when we need to calibrate the entire list
    return quality_scores * calibration_values.w + calibration_values.b

  w_values = np.where(quality_scores > calibration_values.threshold,
                      calibration_values.w, 1.0)
  b_values = np.where(quality_scores > calibration_values.threshold,
                      calibration_values.b, 0.0)
  return quality_scores * w_values + b_values
