## Chinese NER using Bert+LSTM+CRF


### requirement

transformers>=4.6.1
### input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```

### run the code

1. python3 data_convert.py
2. `sh scripts/run_ner_crf_lstm.sh`
