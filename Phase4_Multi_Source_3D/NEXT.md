# NEXT: remaining actions (ordered)

1. **H20 production gate:** run `RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md` §1–9 on the 16 × H20 server.
   * Environment build (`setup_env.sh`; offline: `R3D_DEPS`, `R3D_WHEELS`).
   * Regression suites, Gate A × 3 builds, cross-device bit-exactness.
   * A ≥ 256-scene canary under contract v3, FastLabel vs exact tau-pencil, crash/restart.
   * Throughput, then the 100K projection with `scripts/estimate.py`.

   None of this has run on an H20.
2. **κ / coverage review** (H20 runbook §7, PI decision). Freeze κ = (0.15, 0.15, 0.15, 0.55) and the L64 coverage
   slot, or change them and bump the contract id. The laptop canary found T* ESS fractions of 0.78 / 0.87 / 0.94 at
   L = 64 / 128 / 256, and a max/median weight of 2.6 at L64 (`RELEASE_CANARY_2026-11-18.md`).
3. **Code freeze for the campaign.**
   * Tag the production commit and run every node from it.
   * `CODE.json` pins the source digest per root. Resuming under different code needs `--allow-code-change`, and
     each such attempt counts against the union bound (0.01 holds up to 7.3·10⁶ attempts).
   * Never re-roll a failed statistical certificate with another seed (the contract pins `base_seed`).
4. **Authorization:** the 100K campaign needs separate explicit approval. Command: `PRODUCTION_RUNBOOK.md` §3, with
   `--slots 0:100000` spread over nodes.
5. **Loader trust (H2 of the FastLabel review):** the loader assumes `verify_root.py --labels` ran. Run it on every
   finished root and keep its `VERIFY.json` with the dataset manifest.
6. **Optional strengthening:**
   * exact enclosures of B's cell volumes (they are binary64 today; see `FASTLABEL_V34_CERTIFICATION.md` §6);
   * an independent replay of a random audit sample of released labels.
7. **Research (not blocking):**
   * XLF deterministic solver: fix R1, then the near-occluder-edge stall (`RESEARCH_ARCHIVE.md`).
   * Certified labels for curved occluders (`docs/CURVED_ASSESSMENT.md`).
