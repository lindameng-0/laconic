# Two-agent toy example

A minimal researcher → writer workflow with Laconic on the seam. Runs fully
offline (the "agents" are scripted), so the token report you see is real but
free.

```bash
python run.py                          # conservative strategy, gpt-4.1 tokenizer
python run.py --strategy off           # measurement only — profile before compressing
python run.py --strategy aggressive    # budgeted pruning
python run.py --html report.html       # also write the HTML report
```

What to look for in the output:

- The `researcher→writer` edge aggregates three handoffs.
- Handoffs 2 and 3 repeat the appendix block — dedup replaces it with a short
  reference (`dedup hit`), which is where the biggest saving comes from.
- With `--strategy off` nothing changes and the table shows the baseline burn.
- Token counts are exact if `tiktoken` is installed (the `gpt-4.1` default);
  otherwise they are labeled estimates.
