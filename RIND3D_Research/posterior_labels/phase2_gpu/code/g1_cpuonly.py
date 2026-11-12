import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl import run_p2, run_g1
for f in sys.argv[1:]:
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    run_p2.run_window(g, w, d1=4); run_g1.run_window(g, w, d1=4)
    for name, fn in (('p2', run_p2.run_window), ('g1cpu', run_g1.run_window), ('p2', run_p2.run_window), ('g1cpu', run_g1.run_window)):
        c = time.process_time(); r = fn(g, w, d1=4); print(os.path.basename(f), name, '%.2f' % (time.process_time() - c), flush=True)
