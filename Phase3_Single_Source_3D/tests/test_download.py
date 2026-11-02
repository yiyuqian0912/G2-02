import io, tarfile
from pathlib import Path
import pytest
import zstandard as zstd
from rind3d_reader.catalog import Catalog, FORMAT
from rind3d_reader.releases import seal_release, pack_release, sha_file
from rind_phase3 import download


@pytest.fixture
def transport(tmp_path):
    # Zero-scene metadata fixture, explicitly not scientific/READY data.
    root = tmp_path / "synthetic-transport"
    root.mkdir()
    with Catalog(root, create=True):
        pass
    (root / "catalog/.writer.lock").unlink()
    (root / "README.md").write_text("Synthetic transport unit test, not a dataset\n")
    (root / "splits").mkdir()
    for split in ("train", "val", "test"):
        (root / "splits" / f"{split}.txt").write_text("")
    m = seal_release(
        root,
        {
            "format": "RIND3D",
            "version": 2,
            "name": "synthetic-transport",
            "kind": "full",
            "catalog_format": FORMAT,
            "privacy_policy": "full-field-training-v1",
            "scene_count": 0,
            "window_count": 0,
            "scene_records": [],
            "included_scene_ids": [],
            "not_scientific_data": True,
        },
    )
    packed = pack_release(root, tmp_path / "unit.tar.zst")
    spec = {
        "release_version": "unit",
        "archive_name": "RIND3D-unit.tar.zst",
        "archive_bytes": packed["bytes"],
        "archive_sha256": packed["sha256"],
        "release_id": m["release_id"],
        "drive_file_id": "synthetic-id",
    }
    return Path(packed["archive"]), spec


def test_stream_verify_atomic_install_and_idempotence(tmp_path, transport, monkeypatch):
    archive, spec = transport
    calls = []

    def response(_):
        calls.append(True)
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(download, "_drive_response", response)
    root = download.install(tmp_path / "data", spec=spec)
    assert root.name == "RIND3D-vunit" and (root / "manifest.json").is_file()
    assert download.install(tmp_path / "data", spec=spec) == root and len(calls) == 1
    assert not list((tmp_path / "data").glob(".rind3d-extract-*"))


def test_corrupt_download_not_published(tmp_path, transport, monkeypatch):
    _, spec = transport
    monkeypatch.setattr(download, "_drive_response", lambda _: io.BytesIO(b"corrupted"))
    with pytest.raises(ValueError, match="SHA256"):
        download.install(tmp_path / "data", spec=spec)
    assert not (tmp_path / "data" / "RIND3D-vunit").exists()
    assert not list((tmp_path / "data").glob(".rind3d-download-*"))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "root/link"])
def test_archive_traversal_and_symlink_refused(tmp_path, transport, name):
    _, spec = transport
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        item = tarfile.TarInfo(name)
        if name.endswith("link"):
            item.type = tarfile.SYMTYPE
            item.linkname = "../outside"
        else:
            item.size = 1
        tar.addfile(item, io.BytesIO(b"x"))
    archive = tmp_path / "unsafe.tar.zst"
    archive.write_bytes(zstd.ZstdCompressor().compress(raw.getvalue()))
    spec = {
        **spec,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha_file(archive),
    }
    with pytest.raises(ValueError):
        download.install(tmp_path / "data", archive=archive, spec=spec)
    assert not (tmp_path / "escape").exists() and not (tmp_path / "absolute").exists()


def test_existing_different_data_never_overwritten(tmp_path, transport):
    _, spec = transport
    output = tmp_path / "data"
    existing = output / "RIND3D-vunit"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("retain")
    with pytest.raises((ValueError, FileNotFoundError)):
        download.install(output, spec=spec)
    assert (existing / "keep.txt").read_text() == "retain"


def test_drive_confirmation_form_fields_are_preserved():
    parser = download.Confirmation()
    parser.feed(
        '<form action="https://drive.usercontent.google.com/download"><input name="id" value="abc"><input name="confirm" value="t"><input name="uuid" value="token"></form>'
    )
    assert dict(parser.forms[0]["fields"]) == {
        "id": "abc",
        "confirm": "t",
        "uuid": "token",
    }
