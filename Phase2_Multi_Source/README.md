# RIND Phase II — Multi-source localization

[中文说明](README.zh-CN.md) · [Provenance](docs/upstream.md) · [Method](docs/method.md) · [Interfaces](docs/interfaces.md)

Imported from Yushu's `yushu/phase1-multi-source` branch, retaining the Phase I module layout in a sibling project. Package: `rind_phase2`; dataset class: `Phase2Dataset`.

Working modules: installation, additive observations, data browser, joint-source rerendering, joint candidate sampling, physical teacher and diagnostics. The release describes 1–4 random-strength sources, additive response and variable-count source-free quadtree windows. Source intensities and count uncertainty are part of the inverse hypotheses.

The original joint-posterior student remains a placeholder. The independent `research/layered_residual/` path now provides layered teachers, a single combined full-field student target, training, reports, and an interactive test-set browser. This field is a contribution-location distribution, not a joint multi-source posterior.

```bash
uv sync --locked --extra train --extra browser --extra diagnostics
PYTHONPATH=.:src .venv/bin/python -m unittest discover -s tests -q
./run_training.sh --check
```

Install the multi-source ZIP with `./scripts/install_data.sh PATH --keep-archive`; open the raw data browser with `./scripts/browser.sh` (localhost:8768). `RIND_MULTI_DATA_ROOT` can point to an existing multi-source installation. See the Chinese guide for responsibilities and next steps.

After layered training, run `./view_results.sh` and open `http://127.0.0.1:8768` to browse observations, teacher/student fields, and every residual peel.
