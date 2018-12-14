Seq2seq learning for English to Katakana and argsort. Code is based on https://wanasit.github.io/

## English to Katakana

In `en2jp.py`, a basic seq2seq model and a seq2seq model with attention are included. To run:

```bash
python3 en2jp.py
```

## Argsort

`ptr.py` contains a seq2seq model, a seq2seq model with attention, and a pointer network. To run:

```bash
python3 ptr.py
```

To generate data for argsort:

```bash
python3 generate_data.py
```

