"""Small stdlib PEP 517 backend for this pure Python reader package."""
import base64,hashlib,io,csv,tarfile,zipfile
from pathlib import Path

NAME='rind3d_reader';VERSION='0.1.0';DIST=f'{NAME}-{VERSION}.dist-info'
META='''Metadata-Version: 2.2
Name: rind3d-reader
Version: 0.1.0
Summary: Reader-only SDK for RIND3D
Requires-Python: >=3.12,<3.14
Requires-Dist: numpy>=2.2,<3
Requires-Dist: zstandard>=0.23
Provides-Extra: train
Requires-Dist: torch>=2.4; extra == "train"
Provides-Extra: legacy
Requires-Dist: pyarrow>=20,<26; extra == "legacy"

License and citation metadata are pending. No license grant is implied.
'''
def get_requires_for_build_wheel(config_settings=None):return []
def prepare_metadata_for_build_wheel(metadata_directory,config_settings=None):
 p=Path(metadata_directory)/DIST;p.mkdir(parents=True,exist_ok=True);(p/'METADATA').write_text(META);return DIST
def build_wheel(wheel_directory,config_settings=None,metadata_directory=None):
 root=Path(__file__).parent;name=f'{NAME}-{VERSION}-py3-none-any.whl';files={str(p.relative_to(root/'src')):p.read_bytes() for p in (root/'src').rglob('*.py')}
 files[DIST+'/METADATA']=META.encode();files[DIST+'/WHEEL']=b'Wheel-Version: 1.0\nGenerator: rind3d-reader-stdlib\nRoot-Is-Purelib: true\nTag: py3-none-any\n';files[DIST+'/entry_points.txt']=b'[console_scripts]\nrind3d-reader = rind3d_reader.cli:main\n'
 records=[]
 for path,data in sorted(files.items()):records.append((path,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(),len(data)))
 records.append((DIST+'/RECORD','',''));s=io.StringIO();csv.writer(s,lineterminator='\n').writerows(records);files[DIST+'/RECORD']=s.getvalue().encode()
 Path(wheel_directory).mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(Path(wheel_directory)/name,'w',zipfile.ZIP_DEFLATED) as z:
  for path,data in sorted(files.items()):
   info=zipfile.ZipInfo(path,(2026,1,1,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
 return name
def build_sdist(sdist_directory,config_settings=None):
 root=Path(__file__).parent;name=f'{NAME}-{VERSION}.tar.gz';Path(sdist_directory).mkdir(parents=True,exist_ok=True)
 with tarfile.open(Path(sdist_directory)/name,'w:gz') as t:
  for p in sorted(root.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=f'{NAME}-{VERSION}/'+str(p.relative_to(root)))
 return name
