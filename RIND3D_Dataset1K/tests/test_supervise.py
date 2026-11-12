"""The label supervisor survives native crashes and hangs, and reports each exactly once."""
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from rind3d_1k.supervise import Supervisor  # noqa: E402


def test_crash_hang_and_results():
    tasks = ["a", "abort", "b", "hang", "segv", "c", "d", "e"]
    res, fail = [], []
    sup = Supervisor(mp.get_context("spawn"), ["gpu", "cpu", "cpu"], "_sup_tasks:job", {"x": 1}, nice=0,
                     sys_path=[str(HERE)])
    sup.run(tasks, lambda t: 3.0, res.append, lambda t, r: fail.append((t, r)), poll=0.5)
    assert sorted(r["task"] for r in res) == ["a", "b", "c", "d", "e"]
    assert all(r["kw"] == {"x": 1} for r in res)
    kinds = {t: r["kind"] for t, r in fail}
    assert kinds == {"abort": "crash", "segv": "crash", "hang": "timeout"}
    assert dict(fail)["abort"]["exitcode"] == -6
    assert all(not s["proc"].is_alive() for s in sup.slots)
