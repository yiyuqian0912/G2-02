from types import SimpleNamespace
import numpy as np
from rind_phase3.teacher.octree import (
    initial_cells,
    OctreeTeacher,
    VariationRefinement,
)


class Evaluator:
    probes = SimpleNamespace(observed=np.ones(4))

    def evaluate(self, p):
        valid = p[:, 0] > 0.1
        return np.where(valid, p[:, 0], np.nan), SimpleNamespace(
            evaluated_pairs=int(valid.sum()) * 4,
            validity=SimpleNamespace(
                valid=valid,
                status=tuple(
                    "CERTIFIED_VALID" if v else "CERTIFIED_INVALID" for v in valid
                ),
                reasons=tuple("fixture" for _ in valid),
            ),
        )


def test_eight_512_initial_cubes_and_exact_children():
    cells = initial_cells()
    assert len(cells) == 8 and {c.side for c in cells} == {512}
    assert {c.origin for c in cells} == {
        (x, y, z) for x in (0, 512) for y in (0, 512) for z in (0, 512)
    }
    assert sum(c.volume for c in cells) == 1
    children = cells[0].children()
    assert len(children) == 8 and all(c.side == 256 for c in children)
    assert sum(c.volume for c in children) == cells[0].volume


def test_deterministic_constant_k_and_no_homogeneity_claim():
    a = OctreeTeacher(Evaluator(), k=4, seed=11, max_pairs=256).run(1)
    b = OctreeTeacher(Evaluator(), k=4, seed=11, max_pairs=256).run(1)
    assert (
        a["trace"] == b["trace"]
        and a["candidate_attempts"] == 64
        and a["cells_refined"] == 1
    )
    assert a["physical_source_receiver_pairs"] <= 256 and all(
        len(r["sample_xyz"]) == 4 for r in a["trace"]
    )
    assert (
        not a["teacher_support_complete"] and not a["source_space_coverage_certified"]
    )
    assert all(
        r["region_certification"] == "UNRESOLVED" and not r["pruned"]
        for r in a["trace"]
    )


def test_insufficient_budget_is_explicit_and_keeps_unresolved_regions():
    r = OctreeTeacher(Evaluator(), max_candidates=8, max_pairs=32).run(1)
    assert not r["execution_complete"] and r["candidate_attempts"] == 8
    assert (
        len(r["pending_unresolved_cells"]) == 6
        and r["termination"] == "query_budget_incomplete"
    )


def test_variation_refinement_has_explicit_exploration():
    result = OctreeTeacher(
        Evaluator(), refinement=VariationRefinement(exploration=1), max_pairs=256
    ).run(1)
    assert (
        result["refinement_policy"] == "VariationRefinement"
        and result["cells_refined"] == 1
    )
