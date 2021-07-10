# DeepConsensus Test Data

To regenerate the files under `learning/genomics/deepconsensus/testdata/output`,
run the below command. You will need to do this if you have modified the data
processing code in a way that changes the outputs from each Beam pipeline
(`merge_datasets`, `generate_input`, `write_tf_examples`).

```
time ./learning/genomics/deepconsensus/scripts/generate_testdata.sh
```

This command should take ~6 min to complete.
