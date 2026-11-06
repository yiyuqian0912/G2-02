"""Build into a NEW angular experiment directory; never overwrite old binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sources():
    here=Path(__file__).parent
    return {str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for d in (here,here.parent/'b2_native') for p in sorted(d.iterdir())
            if p.suffix in ('.cpp','.hpp') or p.name=='CMakeLists.txt'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/gaussian-b2-angular-v1/build'))
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cmake=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip()
    subprocess.run(['cmake','-S',str(Path(__file__).parent),'-B',str(out),f'-DPython_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={cmake}','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(out),'--parallel','2'],check=True)
    binaries=list(out.glob('_b2_angular*.so'));assert len(binaries)==1
    identity={'source_sha256':sources(),'extension_sha256':hashlib.sha256(binaries[0].read_bytes()).hexdigest(),
              'python':sys.version,'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
              'flags':['-fno-fast-math','-ffp-contract=off'],'ready_capable':False}
    (out/'build-identity.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':main()
