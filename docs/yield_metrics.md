# Yield improvement versus ccs on various sequencing runs

## We evaluate on 3 different datasets

For each PacBio dataset (Movie ID), we compared yield at Q30 for ccs (baseline),
and v0.2, v0.3, v1.0, v1.1 of DeepConsensus.

Movie ID             | Sample | Chemistry | Mean insert size
-------------------- | ------ | --------- | ----------------
m64011_181218_235052 | HG002  | 1         | 11 kb
m64008_201124_002822 | HG002  | 2.2       | 15 kb
m64014_200920_132517 | HG002  | 2.2       | 24 kb

## Yield versus runtime

![v1.0 runtime versus yield over ccs](images/runtime_yield.png)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>version</th>
      <th>movie</th>
      <th>dataset</th>
      <th>num_reads_ccs</th>
      <th>num_reads</th>
      <th>yield@emQ20</th>
      <th>yield@emQ20/ccs</th>
      <th>yield@emQ30</th>
      <th>yield@emQ30/ccs</th>
      <th>yield@emQ40</th>
      <th>yield@emQ40/ccs</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>v1.1</td>
      <td>m64011_181218_235052</td>
      <td>chem1_11kb</td>
      <td>1,392,300</td>
      <td>1,557,424</td>
      <td>17.18 Gb</td>
      <td>111.83%</td>
      <td>12.14 Gb</td>
      <td>137.40%</td>
      <td>5.10 Gb</td>
      <td>208.66%</td>
      <td>233.02</td>
    </tr>
    <tr>
      <td>v1.1</td>
      <td>m64008_201124_002822</td>
      <td>chem2.2_15kb</td>
      <td>2,687,977</td>
      <td>2,899,794</td>
      <td>42.97 Gb</td>
      <td>108.49%</td>
      <td>32.74 Gb</td>
      <td>128.44%</td>
      <td>9.64 Gb</td>
      <td>241.85%</td>
      <td>567.41</td>
    </tr>
    <tr>
      <td>v1.1</td>
      <td>m64014_200920_132517</td>
      <td>chem2.2_24kb</td>
      <td>1,918,627</td>
      <td>2,087,945</td>
      <td>49.74 Gb</td>
      <td>109.94%</td>
      <td>32.52 Gb</td>
      <td>194.41%</td>
      <td>2.73 Gb</td>
      <td>1058.0%</td>
      <td>724.48</td>
    </tr>
  </tbody>
</table>

`yield@emQ30/ccs` or "Yield at empirical Q30 relative to CCS" is calculated as
follows:

1.  Filter DeepConsensus output to predicted Q20.
2.  For each read, align it to the truth and calculate identity from that
    alignment: identity = # matches / (# matches + # mismatches + #
    insertions + # deletions).
3.  Take all the reads that have identity >= 0.999 (this is Q30).
4.  Because longer reads are more useful than shorter reads, we count the total
    bases and not just the number of reads.
5.  Next we repeat the above for the original CCS reads (run with default
    params = Q20 filtered) and subtract and divide them to get a percentage,
    e.g. 40% percent means that DeepConsensus increased yield of high quality
    reads in bases by 40% over CCS.

These were run on GCP `n1-standard-16` machines with no GPU (in 500 shards,
combined above), with `--batch_zmws=100 --batch_size=1024`, which is generally
what we recommend. For more information on compute setups, see the
[runtime metrics page](runtime_metrics.md).

## Runtime-yield tradeoffs with `--skip_windows_above`

The `--skip_windows_above` option (introduced in v0.3) allows DeepConsensus to
skip windows whose average CCS base qualities are already above a certain
quality threshold. The windows that are skipped just adopt the CCS sequence
without correction. This saves runtime, but there is a yield tradeoff, shown in
this chart for m64014_200920_132517-chr20:

![runtime/yield tradeoff of --skip_windows_above](images/skip_windows_above_tradeoff.png).

The default in v1.0 is Q45, but you can adjust this level using
`--skip_windows_above`.
