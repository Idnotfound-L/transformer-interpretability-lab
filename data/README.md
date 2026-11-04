# Data

The code's built-in text is a tiny synthetic/repetitive smoke-test corpus, not a research benchmark. For a more credible local-model demonstration, place a UTF-8 plain-text corpus here (for example, a public-domain literary corpus whose exact redistribution terms you have checked) and train with:

```bash
python scripts/train_tiny.py --corpus data/my_corpus.txt --steps 5000 --block-size 128 --seed 42
```

Do not commit private, licensed or otherwise restricted corpora. Record the source, licence, preprocessing and train/validation split in your report. The trainer stores the vocabulary and training metadata in the checkpoint.
