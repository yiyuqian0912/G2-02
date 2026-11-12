# Incident: test-split geometry read by a unit test (2026-11-11)

The first run of `tests/test_tau_lemmas.py` took the first three scene ids in sorted order without filtering
by split. Two of them (0097daf4…, 00d66177…) belong to the **test** split.

**What was read.** `rpl.data.Scene()` read their geometry and metadata, and also their stored source position
and window records.

**How they were used.** Only the geometry rows were used, for 200,000 random segment-predicate pairs per
scene. That checks Lemma 2/2W against the Cyrus–Beck predicate. No window response was read, no label or
posterior was computed for those scenes, and nothing about their sources or windows enters any result.

**Remedy.**
* The test now filters to train scenes and was rerun; `outputs/TEST_TAU_LEMMAS.json` holds the rerun on
  train scenes only.
* All reported real-window results use the validation split (real-window batches and selection study) or
  train scenes (development).
