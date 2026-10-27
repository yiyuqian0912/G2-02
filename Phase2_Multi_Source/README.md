# RIND Phase II — Multi-source localization

[中文说明](README.zh-CN.md) · [Provenance](docs/upstream.md) · [Method](docs/method.md) · [Interfaces](docs/interfaces.md)

Imported from Yushu's `yushu/phase1-multi-source` branch, retaining the Phase I module layout in a sibling project. Package: `rind_phase2`; dataset class: `Phase2Dataset`.

## Dataset versions

The completed **100,000-scene unit-strength release** uses **256 informative random windows per scene** (25.6 million observations), sizes **64/128/256/512**, and random positions. All sources lie outside the observation. Constant responses are excluded over free pixels, including constant intermediate intensities. The ZIP is **1.55 GiB**, and installed data uses **62.41 GiB**. The original 10k random-strength quadtree release remains supported; the manifest distinguishes the formats.

Use the same `./scripts/install_data.sh`, `./scripts/browser.sh` and `Phase2Dataset`. See the [new dataset guide](docs/dataset-unit-random-100k.md) for sampling rules, compact per-source storage, reference solutions, installation, DataLoader usage and build status. The new Drive link will be added after upload; existing links refer to the old release.

These calls require this updated Phase II code, including its vendored reader. No separate companion project or new dependency is required. The dataset ZIP does not update an older checkout's code automatically.

Working modules: installation, additive observations, data browser, joint-source rerendering, joint candidate sampling, physical teacher and diagnostics. The original release describes 1–4 random-strength sources and source-free quadtree windows. For the new release, the joint teacher reads the manifest's unit-strength policy; source counts and count priors remain explicit caller settings.

The original joint-posterior student remains a placeholder. The independent `research/layered_residual/` path now provides layered teachers, a single combined full-field student target, training, reports, and an interactive test-set browser. This field is a contribution-location distribution, not a joint multi-source posterior.

```bash
uv sync --locked --extra train --extra browser --extra diagnostics
PYTHONPATH=.:src:../Phase1_Single_Source_Single_Intensity/src .venv/bin/python -m unittest discover -s tests -q
./run_training.sh --check
```

Install the multi-source ZIP with `./scripts/install_data.sh PATH --keep-archive`; open the raw data browser with `./scripts/browser.sh` (localhost:8768). `RIND_MULTI_DATA_ROOT` can point to an existing multi-source installation. See the Chinese guide for responsibilities and next steps.

After layered training, run `./view_results.sh` and open `http://127.0.0.1:8768` to browse observations, teacher/student fields, and every residual peel.
