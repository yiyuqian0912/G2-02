"""Write EVIDENCE_MANIFEST.json (SHA256 of every study file except the venv and caches) and the review ZIP,
then verify every ZIP entry against the manifest."""
import hashlib, json, os, sys, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.venv', '__pycache__', '.pytest_cache'}

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def files():
    for d, ds, fs in os.walk(ROOT):
        ds[:] = sorted(x for x in ds if x not in SKIP_DIRS)
        for f in sorted(fs):
            p = os.path.join(d, f); r = os.path.relpath(p, ROOT)
            if r in ('EVIDENCE_MANIFEST.json',) or f.endswith(('.nbi', '.nbc', '.pyc')):
                continue
            yield r, p

def main(zip_path, max_file_bytes=20_000_000):
    man = {r: dict(sha256=sha(p), bytes=os.path.getsize(p)) for r, p in files()}
    json.dump(dict(note='SHA256 of every file of the study except this manifest, .venv and caches', files=man),
              open(os.path.join(ROOT, 'EVIDENCE_MANIFEST.json'), 'w'), indent=1)
    rels = [r for r in man if man[r]['bytes'] <= max_file_bytes] + ['EVIDENCE_MANIFEST.json']
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for r in rels:
            z.write(os.path.join(ROOT, r), 'RIND3D_POSTERIOR_LABELS_REVIEW/' + r)
    bad = []
    with zipfile.ZipFile(zip_path) as z:
        for r in rels:
            if r == 'EVIDENCE_MANIFEST.json':
                continue
            if hashlib.sha256(z.read('RIND3D_POSTERIOR_LABELS_REVIEW/' + r)).hexdigest() != man[r]['sha256']:
                bad.append(r)
    skipped = [r for r in man if man[r]['bytes'] > max_file_bytes]
    print(json.dumps(dict(entries=len(rels), bytes=os.path.getsize(zip_path), sha256=sha(zip_path), mismatches=bad,
                          skipped_large=skipped)))

if __name__ == '__main__':
    main(sys.argv[1])
