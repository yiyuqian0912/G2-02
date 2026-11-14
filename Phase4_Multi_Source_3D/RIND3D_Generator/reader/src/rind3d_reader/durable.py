"""POSIX durable writes. Publication never overwrites an immutable record."""
import contextlib,hashlib,json,os,tempfile
from pathlib import Path

def canonical(value):
 normalized=json.loads(json.dumps(value,allow_nan=False))
 return json.dumps(normalized,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_bytes(path,data,*,replace=True):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=None
 try:
  with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
   temporary=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
  if replace:os.replace(temporary,path)
  else:os.link(temporary,path)
  fsync_dir(path.parent)
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)
def write_json(path,value,*,replace=True):write_bytes(path,canonical(value)+b'\n',replace=replace)
def seal(value):return {'payload':value,'sha256':digest(value)}
def read_sealed(path):
 record=json.loads(Path(path).read_text())
 if set(record)!= {'payload','sha256'} or digest(record['payload'])!=record['sha256']:raise ValueError('Record checksum mismatch: '+str(path))
 return record['payload']
@contextlib.contextmanager
def lock(path,*,shared=False,blocking=False):
 import fcntl
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+b') as f:
  try:fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|(0 if blocking else fcntl.LOCK_NB))
  except BlockingIOError:raise RuntimeError('Active owner/writer: '+str(path)) from None
  try:yield f
  finally:fcntl.flock(f,fcntl.LOCK_UN)
