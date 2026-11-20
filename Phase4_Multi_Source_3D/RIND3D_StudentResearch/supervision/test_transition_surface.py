"""Fence privileged writers. Changes require a new implementation-correspondence review."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]/"RIND3D_Posterior/experiments/credit_solver/cbc_kernel.py"


def test_kernel_transition_surface():
    tree = ast.parse(SOURCE.read_text())
    kernel, = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Kernel"]
    commits, credit_writers, writes = {}, set(), {}
    for fn in kernel.body:
        if not isinstance(fn,ast.FunctionDef):
            continue
        for n in ast.walk(fn):
            if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and n.attr in {"_state","_best","_tainted"}:
                writes.setdefault(n.attr,set()).add(fn.name)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="_commit":
                commits[fn.name] = commits.get(fn.name,0)+1
                if len(n.args)>2 or any(k.arg!="exclude" for k in n.keywords):
                    credit_writers.add(fn.name)
    assert commits == dict(exclude_window=1,condition_pair=1,split=1,projective_patch=2,screen=2,
                           _settle=2,enforce=1)
    assert credit_writers == {"_settle"}
    assert writes == {"_state":{"__init__","_commit"},"_best":{"__init__","snapshot"},"_tainted":{"__init__"}}
    guard, = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_transition"]
    assert any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and n.attr=="_tainted"
               for n in ast.walk(guard))
