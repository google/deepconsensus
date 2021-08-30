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
"""Utility functions being used for data processing."""


def get_pacbio_molecule_name(name):
  """Returns the molecule name from the full PacBio name.

  Args:
    name: str. fragment name or reference name from PacBio subreads, CCS reads,
      or truth reads.  For PacBio data, the name is of the format
      '<movieName>/<zmw>/<indices_or_type>'. We remove the '/<indices_or_type>'
      suffix to produce the molecule name. <indices_or_type> is different
      depending on the type of PacBio reads. For subreads, the suffix is of the
      form '<qStart>_<qEnd>', where <qStart> and <qEnd> are the indices into the
      polymerase read at which the subread starts and ends. For CCS reads, the
      suffix is 'ccs'. For truth reads, the suffix is 'truth'.
  """

  split_name = (name.split('/'))

  # This function can be called with a reference name. When reads are unmapped,
  # `name` is empty, and we won't be able to extract the molecule name.
  if len(split_name) != 3:
    return None
  return str('/'.join(split_name[:2]))
