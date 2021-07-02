"""Tests for deepconsensus .postprocess.plx_ingest."""

import datetime
import glob

from absl.testing import absltest
import pandas as pd

from deepconsensus.postprocess import plx_ingest

TEST_PATH = '/cns/is-d/home/brain-genomics/danielecook/deepconsensus/experiments/20210331/exp_23492584/wu_27/postprocess/20210331052812'


class ParsePathsTest(absltest.TestCase):

  def test_parse_paths(self):
    ds = plx_ingest.parse_paths(TEST_PATH)
    self.assertEqual(ds['work_unit'], 'wu_27')
    self.assertEqual(ds['xid'], 23492584)
    self.assertEqual(ds['id'], '20210331052812')


class LoadDataSetTest(absltest.TestCase):

  def test_load_dataset(self):
    ds = plx_ingest.parse_paths(TEST_PATH)
    fname = 'identity_metrics.csv'
    for fname in plx_ingest.OUTPUTS_TO_PLX:
      ds[fname] = True
      combined_dataset = plx_ingest.load_datasets([ds, ds, ds], fname)
      self.assertNotEmpty(combined_dataset)


class SaveDatasetTest(absltest.TestCase):

  def setUp(self):
    super(SaveDatasetTest, self).setUp()
    plx_ingest.DEEPCONSENSUS_PATH = self.create_tempdir()

  def test_save_dataset(self):
    df = pd.DataFrame({'A': [1, 2, 3]})
    plx_ingest.save_dataset(df, 'test_dataset.csv')
    fpath = glob.glob(plx_ingest.DEEPCONSENSUS_PATH.full_path + '/*')
    self.assertLen(fpath, 1)


class FetchPerformanceTest(absltest.TestCase):

  def test_fetch_experiment_info(self):
    self.assertNotEmpty(plx_ingest.fetch_performance(testing=True))


class FetchExperimentTest(absltest.TestCase):

  def test_fetch_experiment_info(self):
    xid = 20825069
    experiment = plx_ingest.fetch_experiment(xid)
    creation_date = datetime.datetime(2021, 2, 15, 15, 16, 59, 599545)
    self.assertEqual(experiment['xname'], 'oed_exp_20210215')
    self.assertEqual(experiment['creation_date'], creation_date)
    self.assertEqual(experiment['citc'], 'dc_sweep')
    self.assertEqual(experiment['notes'], 'xentropy + one-sided edit distance.')
    self.assertEqual(experiment['cl'], 357563957)


if __name__ == '__main__':
  absltest.main()
