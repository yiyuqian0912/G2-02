"""Time the entire command including initialization, validation and waited-for children (Linux)."""
import json
import os
from pathlib import Path
import sys
from time import monotonic

dest=Path(sys.argv[1])
if dest.exists():
    raise FileExistsError(dest)
start=monotonic()
pid=os.posix_spawnp(sys.argv[2],sys.argv[2:],os.environ)
_,status,usage=os.wait4(pid,0)
rc=os.waitstatus_to_exitcode(status)
data=dict(wall_s=monotonic()-start,user_s=usage.ru_utime,sys_s=usage.ru_stime,
          max_single_process_rss_kib=usage.ru_maxrss,returncode=rc)
with dest.open("x") as stream:
    json.dump(data,stream)
sys.exit(rc if rc>=0 else 128-rc)
