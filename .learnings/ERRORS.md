# Error Log

## [ERR-20260607-001] huggingface_tokenizer_online_check

**Logged**: 2026-06-07T20:48:12+08:00
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
Local small-LM generation evaluation can fail before generation if
`AutoTokenizer.from_pretrained("gpt2")` performs a Hugging Face online check and
the remote server disconnects.

### Error
```text
httpx.RemoteProtocolError: Server disconnected without sending a response.
```

### Context
- Command/operation attempted: MPS generation evaluation with
  `examples.evaluate_small_lm_generations`.
- The checkpoint and tokenizer files were already available locally, but
  `transformers` still attempted a Hugging Face Hub request while listing repo
  templates.

### Suggested Fix
For local cached evaluations, run with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Consider adding a script-level
`local_files_only=True` tokenizer load path for archived local checkpoints.

### Metadata
- Reproducible: unknown
- Related Files: `examples/evaluate_small_lm_generations.py`
- See Also: `docs/runbooks/tinystories_v61_seq384_recovery.md`

---
