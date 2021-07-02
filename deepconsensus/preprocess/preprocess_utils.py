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
