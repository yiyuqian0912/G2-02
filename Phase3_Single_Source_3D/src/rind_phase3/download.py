"""Stream public RIND3D, verify SHA256, safely extract and atomically install."""

import argparse
from contextlib import contextmanager
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import importlib.resources
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

from rind3d_reader.releases import extract_release, verify_release, sha_file


def release_spec():
    return json.loads(
        importlib.resources.files("rind_phase3").joinpath("release.json").read_text()
    )


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self.current = {"action": a.get("action", ""), "fields": []}
            self.forms.append(self.current)
        elif tag == "input" and self.current is not None and a.get("name"):
            self.current["fields"].append((a["name"], a.get("value", "")))

    def handle_endtag(self, tag):
        if tag == "form":
            self.current = None


def _drive_response(file_id):
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": file_id}
    )
    for _ in range(4):
        response = opener.open(
            urllib.request.Request(
                url, headers={"User-Agent": "RIND3D/0.1 public downloader"}
            ),
            timeout=120,
        )
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            return response
        html = response.read(2_000_001)
        response.close()
        if len(html) > 2_000_000:
            raise ValueError("Oversized Drive confirmation page")
        parser = Confirmation()
        parser.feed(html.decode("utf-8", errors="replace"))
        next_url = None
        for form in parser.forms:
            action = urllib.parse.urljoin(url, form["action"])
            parsed = urllib.parse.urlparse(action)
            if parsed.scheme == "https" and parsed.hostname in (
                "drive.google.com",
                "drive.usercontent.google.com",
            ):
                next_url = (
                    action
                    + ("&" if parsed.query else "?")
                    + urllib.parse.urlencode(form["fields"])
                )
                break
        if not next_url:
            raise RuntimeError(
                "Drive did not return a public download; check sharing/quota/authentication"
            )
        url = next_url
    raise RuntimeError("Drive confirmation did not resolve to a binary download")


def download_archive(destination, spec=None):
    spec = spec or release_spec()
    destination = Path(destination)
    if destination.exists():
        if (
            destination.stat().st_size == spec["archive_bytes"]
            and sha_file(destination) == spec["archive_sha256"]
        ):
            return destination
        raise ValueError(
            "Existing archive is not this verified release; preserve it and choose another output directory"
        )
    if not spec.get("drive_file_id"):
        raise RuntimeError(
            "Public Drive URL is pending; use --archive for verified local installation"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".rind3d-download-", dir=destination.parent)
    temporary = Path(name)
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            os.fdopen(fd, "wb") as out,
            _drive_response(spec["drive_file_id"]) as response,
        ):
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > spec["archive_bytes"]:
                    raise ValueError("Download exceeds declared release size")
                out.write(block)
                digest.update(block)
            out.flush()
            os.fsync(out.fileno())
        if (
            total != spec["archive_bytes"]
            or digest.hexdigest() != spec["archive_sha256"]
        ):
            raise ValueError(
                "Incomplete/corrupt RIND3D download: SHA256 or size mismatch"
            )
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@contextmanager
def install_lock(path):
    import fcntl

    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def install(output, *, archive=None, spec=None):
    spec = spec or release_spec()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"RIND3D-v{spec['release_version']}"
    with install_lock(output / ".rind3d-install.lock"):
        if destination.exists():
            r = verify_release(destination, full=True)
            if r["release_id"] != spec["release_id"]:
                raise ValueError("Existing dataset is a different immutable release")
            return destination
        archive = (
            Path(archive).expanduser().resolve()
            if archive
            else download_archive(output / spec["archive_name"], spec)
        )
        if (
            archive.stat().st_size != spec["archive_bytes"]
            or sha_file(archive) != spec["archive_sha256"]
        ):
            raise ValueError("Archive checksum/size mismatch; existing data retained")
        with tempfile.TemporaryDirectory(
            prefix=".rind3d-extract-", dir=output
        ) as temporary:
            extracted = extract_release(
                archive,
                Path(temporary) / "unpack",
                expected_sha256=spec["archive_sha256"],
            )
            if extracted["release_id"] != spec["release_id"]:
                raise ValueError("Wrong scientific release identity")
            source = Path(extracted["root"])
            if destination.exists():
                raise FileExistsError(
                    "Dataset appeared concurrently; refusing overwrite"
                )
            os.rename(source, destination)
        directory = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument(
        "--archive",
        type=Path,
        help="Install a verified local archive, without downloading",
    )
    a = p.parse_args(argv)
    destination = install(a.output, archive=a.archive)
    print(f"RIND3D installed and SHA256 verified: {destination}")
    print(
        "The historical internal directory/manifest name is preserved as scientific provenance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
