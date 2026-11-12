import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_p2 import run_window
z = np.load(sys.argv[1]); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
run_window(g, w, d1=4)                     # warm-up (JIT)
t = time.process_time(); r = run_window(g, w, d1=4); print('cpu', time.process_time() - t, r['stats']['comp_evals'])
