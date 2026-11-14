# Immutable and incremental releases

A campaign can grow across laptop sessions. A release never changes. Keep scene IDs, geometry IDs, splits, original RINDPack bytes and scientific configuration stable. The release ID is an integrity hash of its canonical manifest; file SHA and brick CRC serve different integrity scopes.

A full snapshot contains all selected READY scenes. A delta contains only new scene directories/catalog records and the metadata needed to describe its complete intended prefix. Every delta names the exact parent release ID. A composed snapshot verifies the parent and creates a new complete directory. Neither the base nor the delta is modified, and scientific file overwrites are refused.

After extending and completing a future campaign prefix, producer commands can create a full snapshot or a delta. Use the frozen producer environment and source-path setup in `PRODUCTION.md`. For example, after separately authorized 2K production:

```bash
.venv/bin/python -m rind3d.campaign_cli export --root /data/core-campaign --output /data/RIND3D-v0.2 --name RIND3D-v0.2 --count 2000 --parent /data/RIND3D-v0.1
.venv/bin/python -m rind3d.campaign_cli export --root /data/core-campaign --output /data/RIND3D-v0.1-to-v0.2 --name RIND3D-v0.1-to-v0.2 --count 2000 --parent /data/RIND3D-v0.1 --delta
```

The reader-only CLI verifies and transports existing snapshots without a generator:

```bash
rind3d-reader verify /data/RIND3D-v0.1
rind3d-reader pack /data/RIND3D-v0.1 /data/RIND3D-v0.1.tar.zst
rind3d-reader extract /data/RIND3D-v0.1.tar.zst /data/fresh-extraction
rind3d-reader apply-delta /data/RIND3D-v0.1 /data/RIND3D-v0.1-to-v0.2 /data/composed-2K
```

Archive entries are ordered with normalized ownership, timestamps and permissions. Archives are streamed with bounded memory and contain no absolute paths, traversal entries, links or special files. A SHA sidecar is written after sealing. RINDPack is already compressed, so outer compression gains may be small. Interrupted archive/extraction work is never treated as a sealed release.

An interrupted archive before its atomic commit leaves only an isolated temporary file; retrying the same still-unused archive name is safe. If the archive exists but its SHA sidecar is missing, retain it for inspection and retry under a new archive name. The packer refuses to overwrite an existing archive. A failed extraction is likewise isolated and must be retried into a new destination. None of these operations edits or regenerates committed scenes.

Delta composition checks full base/delta file integrity, configuration compatibility, parent ID, unchanged prior scene records and unchanged splits. Missing or modified base files are rejected. Composition currently copies scientific files into a new root for isolation; it avoids downloading them again but may need space for the composed copy. Keep base/delta archives or remove redundant extracted copies only after verifying the composition.

A generator, numerical contract or scientific verification change requires explicit versioning and recertification. It is not silently appended as the same distribution. Reader compatibility is recorded in each manifest. SDK 0.1 supports legacy parquet datasets (with PyArrow), immutable catalogs, full snapshots and explicitly composed snapshots; a delta alone is not an independently readable complete dataset.

Checksums do not settle license or attribution. Both remain pending until supplied by the owner. No dataset upload has occurred merely because an archive exists locally.
