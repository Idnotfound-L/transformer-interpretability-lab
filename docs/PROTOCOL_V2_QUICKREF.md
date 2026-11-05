# Protocol V2 — quick reference

## Main runner

```bash
python scripts/run_sequence_copy_study_v2.py --help
```

## Method changes vs V1

- Head discovery scores final-query attention lift to the earlier successor token B; attention to the earlier A is diagnostic only.
- MLP neurons are ranked by clean-vs-corrupted final-query activation sensitivity.
- Random neuron controls are matched by mean absolute activation magnitude.
- Selected components are evaluated on held-out copy prompts and a repetition-free random-transition control.
- Patching includes the earlier successor position, random-position block controls, and component-specific head/neuron patches. The trivial whole-block final-query patch is omitted.
- Per-prompt bootstrap intervals, paired sign-flip tests and BH correction are reported within each run.

See `README.md`, `docs/PROTOCOL_V2_CHANGELOG.md`, and `docs/SEQUENCE_COPY_STUDY.md` for full details.
