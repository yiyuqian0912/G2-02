"""Browse Phase-II residual layers with a v4/v5/v6 Phase-I-style student."""
from __future__ import print_function

import argparse
import base64
import json
from pathlib import Path
import sys
import threading

import numpy as np
from flask import Flask, jsonify, render_template, request


def _pack(array, dtype="<f8"):
    value = np.asarray(array, dtype=dtype)
    return {"shape": list(value.shape), "dtype": dtype,
            "data": base64.b64encode(value.tobytes()).decode()}


def _metrics(q, p):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = q / q.sum()
    p = p / p.sum()
    active_q = q > 0
    active_p = p > 0
    mixture = 0.5 * (q + p)
    entropy = float(-np.sum(q[active_q] * np.log(q[active_q])))
    cross_entropy = float(-np.sum(q[active_q] * np.log(
        np.maximum(p[active_q], np.finfo(np.float64).tiny))))
    js = 0.5 * float(np.sum(q[active_q] * np.log(q[active_q] / mixture[active_q])))
    js += 0.5 * float(np.sum(p[active_p] * np.log(p[active_p] / mixture[active_p])))
    tv = 0.5 * float(np.abs(q - p).sum())
    result = {"cross_entropy": cross_entropy, "teacher_entropy": entropy,
              "forward_kl": cross_entropy - entropy, "js_divergence": js,
              "total_variation": tv, "overlap": 1.0 - tv}
    for mass in (0.5, 0.9):
        teacher_region = _highest_density_region(q, mass)
        student_region = _highest_density_region(p, mass)
        intersection = int(np.count_nonzero(teacher_region & student_region))
        teacher_area = int(np.count_nonzero(teacher_region))
        student_area = int(np.count_nonzero(student_region))
        union = teacher_area + student_area - intersection
        label = str(int(100 * mass))
        precision = intersection / student_area
        recall = intersection / teacher_area
        result.update({
            "support_precision_{}".format(label): float(precision),
            "support_recall_{}".format(label): float(recall),
            "support_dice_{}".format(label): float(
                2 * intersection / (teacher_area + student_area)),
            "support_iou_{}".format(label): float(intersection / union),
            "student_mass_in_teacher_region_{}".format(label): float(p[teacher_region].sum()),
            "teacher_mass_in_student_region_{}".format(label): float(q[student_region].sum()),
            "teacher_region_fraction_{}".format(label): float(teacher_area / len(q)),
            "student_region_fraction_{}".format(label): float(student_area / len(p)),
        })
    return result


def _highest_density_region(probability, mass):
    """Tie-inclusive smallest high-density region containing at least ``mass``."""
    probability = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-probability, kind="stable")
    boundary_index = min(
        int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")),
        len(order) - 1)
    cutoff = probability[order[boundary_index]]
    tolerance = max(np.finfo(np.float64).eps * max(1.0, abs(cutoff)) * 8.0, 1e-15)
    return probability >= cutoff - tolerance


class Explorer:
    def __init__(self, bundle, project, device="cpu", chunk=4096,
                 student_run="layer_student_v4"):
        self.bundle = Path(bundle).resolve()
        self.project = Path(project).resolve()
        self.p2_bundle = self.bundle / "Phase2_Multi_Source"
        self.teacher_root = self.p2_bundle / "outputs/full_training_h20_v2"
        self.student_run = str(student_run)
        self.student_root = self.p2_bundle / "outputs" / self.student_run
        self.chunk = int(chunk)
        phase1 = self.project / "Phase1_Single_Source_Single_Intensity"
        phase2 = self.project / "Phase2_Multi_Source"
        for item in (phase1 / "src", phase2 / "src", phase2 / "vendor/rind-dataset",
                     phase2 / "research/layered_residual"):
            if str(item) not in sys.path:
                sys.path.insert(0, str(item))
        import torch
        from rind_phase1.train import load_model
        from rind_phase2.data import Phase2Dataset
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        data_root = phase2 / "data/raw/phase2-multi-source"
        self.dataset = Phase2Dataset(data_root)
        checkpoint = self.student_root / "models/seed_0/best.pt"
        if not checkpoint.is_file():
            project_checkpoint=(phase2 / "outputs" / self.student_run /
                                "models/seed_0/best.pt")
            if project_checkpoint.is_file():
                checkpoint=project_checkpoint
                self.student_root=project_checkpoint.parents[2]
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "student checkpoint is missing from both the visualization bundle "
                "and project outputs: {}".format(self.student_run))
        self.model, self.checkpoint = load_model(checkpoint)
        self.model.to(self.device).eval()
        self.lock = threading.Lock()
        self.cache = {}
        manifest = json.loads((self.teacher_root / "records.json").read_text())
        lookup = {(int(row["scene_id"]), int(row["view_id"])): row
                  for row in manifest["records"]["test"]}
        entries = []
        for directory in sorted((self.teacher_root / "teachers/test").glob("scene*-view*")):
            report_path = directory / "report.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text())
                identity = (int(report["scene_id"]), int(report["view_id"]))
                row = lookup[identity]
                layers = report.get("layers", [])
                if report.get("status") != "complete" or not layers:
                    continue
                if not all((directory / "layer-{:02d}.npz".format(int(layer["step"]))).is_file()
                           for layer in layers):
                    continue
                entries.append({"index": len(entries), "scene_id": identity[0],
                                "view_id": identity[1], "size": int(row["size"]),
                                "layers": len(layers), "directory": row["directory"]})
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        if not entries:
            raise ValueError("bundle contains no complete teacher windows")
        self.entries = entries

    def catalog(self):
        return {"entries": [{key: value for key, value in row.items() if key != "directory"}
                            for row in self.entries],
                "checkpoint_epoch": int(self.checkpoint["epoch"]),
                "teacher_spacing": 8,
                "target_semantics": "normalized elementwise maximum of per-layer Phase-I fields",
                "note": ("每层分别输入同一个单源学生；顶部学生图是各层学生概率逐点max后归一化。"
                         + (" 学生额外读取完整场景障碍物几何。" if self.model.use_scene_geometry else
                            " 学生不读取完整场景障碍物几何。")),
                "student_run": self.student_run,
                "use_scene_geometry": bool(self.model.use_scene_geometry),
                "device": str(self.device)}

    @staticmethod
    def _raster(values, layer, fill=np.nan):
        image = np.full((len(layer["y_axis"]), len(layer["x_axis"])), fill,
                        dtype=np.float64)
        image.flat[layer["grid_index"]] = values
        return image

    @staticmethod
    def _load_layer(path):
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        required = {"layer_response", "probability", "grid_index", "x_axis", "y_axis",
                    "candidate_xy", "valid", "residual_before", "residual_after"}
        missing = required - set(data)
        if missing:
            raise ValueError("{} missing {}".format(path, sorted(missing)))
        count = len(data["x_axis"]) * len(data["y_axis"])
        xx, yy = np.meshgrid(data["x_axis"], data["y_axis"], indexing="xy")
        candidate = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
        probability = np.zeros(count, dtype=np.float64)
        valid = np.zeros(count, dtype=bool)
        probability[data["grid_index"]] = np.asarray(data["probability"], dtype=np.float64)
        valid[data["grid_index"]] = np.asarray(data["valid"], dtype=bool)
        probability /= probability.sum()
        data.update(candidate_full=candidate, probability_full=probability, valid_full=valid)
        return data

    def _compute(self, index):
        if index in self.cache:
            return self.cache[index]
        if not 0 <= index < len(self.entries):
            raise ValueError("window index out of range")
        import torch
        from rind_phase1.model import student_distribution
        entry = self.entries[index]
        scene, view = entry["scene_id"], entry["view_id"]
        observation = self.dataset.get_observation(scene, view)
        directory = self.teacher_root / entry["directory"]
        report = json.loads((directory / "report.json").read_text())
        layers = []
        for detail in report["layers"]:
            layer = self._load_layer(directory / "layer-{:02d}.npz".format(int(detail["step"])))
            layers.append((detail, layer))
        candidate = layers[0][1]["candidate_full"]
        if any(not np.array_equal(candidate, layer["candidate_full"]) for _, layer in layers):
            raise ValueError("layers do not share one candidate grid")
        responses = np.stack([layer["layer_response"].astype(np.float32) for _, layer in layers])
        obstacles = np.repeat(np.asarray(observation["obstacle"], dtype=bool)[None], len(layers), axis=0)
        windows = np.repeat(np.asarray(observation["window"], dtype=np.float32)[None], len(layers), axis=0)
        candidates = np.repeat(candidate[None], len(layers), axis=0)
        support = np.ones(candidates.shape[:2], dtype=bool)
        geometry=self.dataset.get_obstacle_tokens(scene) if self.model.use_scene_geometry else None
        with self.lock, torch.inference_mode():
            response_t = torch.as_tensor(responses, device=self.device)
            obstacle_t = torch.as_tensor(obstacles, device=self.device)
            window_t = torch.as_tensor(windows, device=self.device)
            candidate_t = torch.as_tensor(candidates, device=self.device)
            support_t = torch.as_tensor(support, device=self.device)
            if geometry is None:
                encoded = self.model.encode_observation(response_t,obstacle_t,window_t)
            else:
                count=len(layers)
                geometry_types=torch.as_tensor(
                    np.repeat(geometry['geometry_types'][None],count,axis=0),
                    dtype=torch.long,device=self.device)
                geometry_params=torch.as_tensor(
                    np.repeat(geometry['geometry_params'][None],count,axis=0),
                    dtype=torch.float32,device=self.device)
                geometry_mask=torch.as_tensor(
                    np.repeat(geometry['geometry_mask'][None],count,axis=0),
                    dtype=torch.bool,device=self.device)
                encoded=self.model.encode_observation(
                    response_t,obstacle_t,window_t,geometry_types,geometry_params,geometry_mask)
            energy = torch.cat([
                self.model.score_encoded(encoded, candidate_t[:, start:start + self.chunk])
                for start in range(0, candidate_t.shape[1], self.chunk)
            ], dim=1)
            _, student_t = student_distribution(energy.float(), support_t)
            student = student_t.cpu().numpy().astype(np.float64)
        student /= student.sum(axis=1, keepdims=True)
        teacher = np.stack([layer["probability_full"] for _, layer in layers])
        teacher_combined = np.max(teacher, axis=0); teacher_combined /= teacher_combined.sum()
        student_combined = np.max(student, axis=0); student_combined /= student_combined.sum()
        result = {"entry": entry, "observation": observation, "report": report,
                  "layers": layers, "teacher": teacher, "student": student,
                  "candidate": candidate, "teacher_combined": teacher_combined,
                  "student_combined": student_combined}
        if len(self.cache) >= 4:
            self.cache.pop(next(iter(self.cache)))
        self.cache[index] = result
        return result

    def result(self, index):
        value = self._compute(index)
        entry = value["entry"]
        observation = value["observation"]
        first = value["layers"][0][1]
        truth = self.dataset.get_source_params(entry["scene_id"])
        layer_meta = []
        for detail, _ in value["layers"]:
            layer_meta.append({"step": int(detail["step"]), "intensity": float(detail["intensity"]),
                               "minimum_physical_cost": detail.get("minimum_physical_cost"),
                               "exact_grid_matches": detail.get("exact_grid_matches"),
                               "conflicts": len(detail.get("trace", {}).get("conflict", []))})
        return {"metadata": {"index": index, "scene_id": entry["scene_id"],
                             "view_id": entry["view_id"],
                             "window": np.asarray(observation["window"]).tolist(),
                             "world_size": int(self.dataset.manifest["global_size"]),
                             "checkpoint_epoch": int(self.checkpoint["epoch"]),
                             "layers": layer_meta,
                             "metrics": _metrics(value["teacher_combined"], value["student_combined"]),
                             "teacher_probability_sum": float(value["teacher_combined"].sum()),
                             "student_probability_sum": float(value["student_combined"].sum()),
                             "student_run":self.student_run,
                             "use_scene_geometry":bool(self.model.use_scene_geometry),
                             "semantics": "per-layer student; normalized max is summary only"},
                "response": _pack(observation["response"]),
                "obstacle": _pack(observation["obstacle"], "|u1"),
                "teacher": _pack(self._raster(value["teacher_combined"], first)),
                "student": _pack(self._raster(value["student_combined"], first)),
                "truth": _pack(truth)}

    def layer(self, index, step):
        value = self._compute(index)
        match = [(position, detail, layer) for position, (detail, layer) in enumerate(value["layers"])
                 if int(detail["step"]) == step]
        if not match:
            raise ValueError("layer step out of range")
        position, detail, layer = match[0]
        obstacle = value["observation"]["obstacle"]
        mask = layer["layer_response"].astype(bool)
        intensity = float(detail["intensity"])
        contribution = intensity * mask.astype(np.float64)
        if not np.allclose(layer["residual_after"], layer["residual_before"] - contribution,
                           rtol=0, atol=1e-10):
            raise ValueError("saved residual peeling equation is inconsistent")
        return {"metadata": {"step": step, "intensity": intensity,
                             "minimum_physical_cost": detail.get("minimum_physical_cost"),
                             "exact_grid_matches": detail.get("exact_grid_matches"),
                             "conflicts": len(detail.get("trace", {}).get("conflict", [])),
                             "metrics": _metrics(value["teacher"][position], value["student"][position])},
                "before": _pack(layer["residual_before"]), "mask": _pack(mask, "|u1"),
                "contribution": _pack(contribution), "after": _pack(layer["residual_after"]),
                "obstacle": _pack(obstacle, "|u1"),
                "probability": _pack(self._raster(value["teacher"][position], layer)),
                "student_probability": _pack(self._raster(value["student"][position], layer))}


def create_app(explorer, template_directory):
    app = Flask(__name__, template_folder=str(template_directory))

    @app.errorhandler(ValueError)
    @app.errorhandler(FileNotFoundError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def home():
        return render_template("phase2_v4_browser.html")

    @app.get("/api/runs")
    def runs():
        return jsonify([{"name": explorer.student_run, "seeds": [0]}])

    @app.get("/api/catalog")
    def catalog():
        return jsonify(explorer.catalog())

    @app.post("/api/result")
    def result():
        payload = request.get_json()
        return jsonify(explorer.result(int(payload.get("index", 0))))

    @app.post("/api/layer")
    def layer():
        payload = request.get_json()
        return jsonify(explorer.layer(int(payload.get("index", 0)), int(payload.get("layer", 1))))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True,
                        help="directory containing Phase1_* and Phase2_Multi_Source")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--student-run",default="layer_student_v4",
                        help=("output directory name, e.g. layer_student_v4, "
                              "layer_student_v5_geometry, or "
                              "layer_student_v6_geometry_residual"))
    args = parser.parse_args()
    explorer = Explorer(args.bundle,args.project,args.device,args.chunk,args.student_run)
    print("Recovered windows: {}".format(len(explorer.entries)), flush=True)
    print("RIND Phase II browser [{}]: http://127.0.0.1:{}".format(
        args.student_run,args.port),flush=True)
    create_app(explorer, Path(__file__).parent / "templates").run(
        host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
