from rind_phase3 import RIND3D


def test_student_whitelist_and_explicit_privileged_geometry(dataset_root):
    with RIND3D(dataset_root, split=None) as data:
        sample = data[0]
        assert set(sample) == {
            "scene_id",
            "window_id",
            "origin",
            "side",
            "observed_response",
            "local_solid_mask",
        }
        assert not hasattr(data, "private_oracle")
        assert not any(
            k in sample
            for k in (
                "structure",
                "geometry",
                "source",
                "candidate_sources",
                "full_response",
                "witnesses",
            )
        )
        teacher = data.teacher_scene(sample["scene_id"])
        assert teacher.geometry_path.name == "geometry.npz" and teacher.geometry_id
