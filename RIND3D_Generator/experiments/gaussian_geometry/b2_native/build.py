"""Build only the isolated research extension; no production build is edited."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sources():
    root=Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
            if p.suffix in ('.hpp','.cpp') or p.name=='CMakeLists.txt'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/gaussian-b2-native-v1/build'));args=p.parse_args()
    root=Path(__file__).parent;out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cmake=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip()
    subprocess.run(['cmake','-S',str(root),'-B',str(out),f'-DPython_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={cmake}','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(out),'--parallel','2'],check=True)
    extension=list(out.glob('_b2_native*.so'))
    if len(extension)!=1:raise RuntimeError('Expected one research extension')
    identity={'source_sha256':sources(),'extension_sha256':hashlib.sha256(extension[0].read_bytes()).hexdigest(),
              'python':sys.version,'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
              'flags':['-fno-fast-math','-ffp-contract=off'],'production_authority':False}
    (out/'build-identity.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':main()
