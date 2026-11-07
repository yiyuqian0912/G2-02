"""Build an isolated maximum-proof library; frozen baselines are never edited."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sources():
    here=Path(__file__).resolve().parent
    return {str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in (here,here.parent/'b2_native')
            for p in sorted(directory.iterdir()) if p.suffix in ('.cpp','.hpp') or p.name=='CMakeLists.txt'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1/build'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cmake=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip()
    subprocess.run(['cmake','-S',str(Path(__file__).resolve().parent),'-B',str(a.output.resolve()),f'-DPython_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={cmake}','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(a.output),'--parallel','2'],check=True)
    binary=list(a.output.glob('_b2_max_cert*.so'));assert len(binary)==1
    identity={'source_SHA256':sources(),'binary_SHA256':hashlib.sha256(binary[0].read_bytes()).hexdigest(),'python':sys.version,
              'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
              'flags':['-fno-fast-math','-ffp-contract=off'],'ready_capable':False}
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':main()
