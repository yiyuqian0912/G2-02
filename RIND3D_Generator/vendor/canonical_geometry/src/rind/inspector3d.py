"""Local slice-based inspector for committed 3D RIND truth snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

from rind.contracts import CSGNode, Leaf, RootKind
from rind.geometry import GeometryView, Membership
from rind.oracle import PrivateOracle
from rind.decay import AdditiveOracle
from rind.publication import validate_snapshot
from rind.storage import read_bundle


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    return parser.parse_known_args()[0]


@st.cache_resource(show_spinner="验证并载入 3D 快照…")
def _load(root: str, snapshot_id: str):
    summary = validate_snapshot(root)
    if summary["snapshot_id"] != snapshot_id or summary["dimension"] != 3:
        raise ValueError("CURRENT 已改变，或快照不是 3D")
    path = Path(root) / "snapshots" / snapshot_id
    bundle = read_bundle(path / "d3" / "core.zarr", 3)
    ledger = pq.read_table(path / "catalogs" / "candidate_ledger.parquet").to_pylist()
    return summary, bundle, ledger


@st.cache_data(show_spinner="计算显示切片…")
def _slice(geometry_id: bytes, snapshot_id: str, z: float, resolution: int,
           _bundle) -> np.ndarray:
    axis = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(),
                              np.full(xx.size, z, dtype=np.float64)))
    result = GeometryView(geometry_id, _bundle).classify_points(points)
    if not np.all(result.valid):
        raise RuntimeError("显示切片存在未证明的点；请查看单点精确查询")
    return result.value.reshape(resolution, resolution)


def _tree(bundle, root_hash: bytes) -> dict:
    item = next(ref for ref in bundle.references if ref.content_hash == root_hash)
    if isinstance(item, CSGNode):
        return {"id": root_hash.hex(), "opcode": item.opcode.name,
                "children": [_tree(bundle, child) for child in item.children]}
    return {"id": root_hash.hex(), "codec": item.codec.name}


def _suggested_slice(bundle, geometry_id: bytes) -> float:
    geometry = next(item for item in bundle.geometries
                    if item.content_hash == geometry_id)
    if geometry.root_kind == RootKind.EMPTY:
        return .5
    refs = {item.content_hash: item for item in bundle.references}
    centers = []
    stack = [geometry.root_hash]
    while stack:
        item = refs[stack.pop()]
        if isinstance(item, CSGNode):
            stack.extend(item.children)
        elif isinstance(item, Leaf):
            centers.append(float(item.field("center").as_array()[2]))
    if not centers:
        return .5
    return min(1., max(0., round(float(np.median(centers)) / .025) * .025))


def _result(result, index: int = 0) -> dict:
    evidence = (result.evidence_ref[index] if hasattr(result, "evidence_ref") else None)
    value = result.value[index]
    work = (result.work[index] if hasattr(result, "work") else
            {"precision_bits": int(result.precision_bits[index])})
    return {"status": int(result.status[index]),
            "value": float(value) if np.issubdtype(result.value.dtype, np.floating)
            else int(value),
            "valid": bool(result.valid[index]),
            "reason_code": result.reason_code[index], "work": work,
            "homogeneous_witness": list(evidence.homogeneous)
            if evidence is not None and hasattr(evidence, "homogeneous") else None}


def _coordinates(label: str, defaults: tuple[float, float, float], key: str) -> list[float]:
    cols = st.columns(3)
    return [cols[i].number_input(f"{label} {axis}", min_value=0.0,
                                 max_value=1.0, value=float(defaults[i]),
                                 format="%.17g", key=f"{key}-{axis}")
            for i, axis in enumerate("xyz")]


def main() -> None:
    st.set_page_config(page_title="RIND 3D 真值查看器", layout="wide")
    st.title("RIND 3D 连续结构真值查看器")
    st.caption("切片是有限采样显示；点、线段、盒体与净空由精确后端判定。")
    root = st.sidebar.text_input("数据集目录", value=str(Path(_args().dataset).resolve()))
    try:
        snapshot_id = (Path(root) / "CURRENT").read_text().strip()
        summary, bundle, ledger = _load(root, snapshot_id)
    except Exception as exc:
        st.error(f"快照无法载入：{exc}")
        st.stop()
    st.sidebar.success(f"快照 {snapshot_id[:12]}")
    st.sidebar.metric("READY 候选", summary["ready_candidates"])
    ready = sorted((row for row in ledger if row["candidate_state"] == "READY"),
                   key=lambda row: row["slot_index"])
    if not ready:
        st.warning("此快照没有 READY 场景。")
        with st.expander("候选记录", expanded=True):
            st.dataframe(ledger)
        return
    selected = st.sidebar.selectbox("候选槽位", ready,
                                    format_func=lambda row: f"槽位 {row['slot_index']:03d} · 变体 {row['scene_variant_index']} · {row['scene_id'][:10]}")
    scene = next(item for item in bundle.scenes
                 if item.content_hash.hex() == selected["scene_id"])
    view = GeometryView(scene.geometry_id, bundle)
    capability = view.capability()
    if not capability["supported"]:
        st.error(f"几何后端不支持此场景：{capability['reason']}")
        st.stop()
    private = st.sidebar.checkbox("显示私有驱动与响应", value=False)
    z = st.sidebar.slider("切片 z", 0.0, 1.0,
                          _suggested_slice(bundle, scene.geometry_id),
                          step=0.025, key=f"slice-{scene.geometry_id.hex()}")
    resolution = st.sidebar.slider("显示采样分辨率", 16, 64, 24, step=8)
    st.info(f"候选 {selected['candidate_id'][:16]} · 尝试 {selected['attempt_index']} · 场景 {selected['scene_id'][:16]}")
    st.caption(f"物理变体：{scene.physics.codec_id}（0 二值，1 固定指数，2 单调 log-PCHIP）")

    left, right = st.columns([1.4, 1])
    with left:
        try:
            grid = _slice(scene.geometry_id, snapshot_id, z, resolution, bundle)
        except Exception as exc:
            st.warning(f"切片显示不可用：{exc}")
        else:
            axis = np.linspace(0, 1, resolution)
            fig = go.Figure(go.Heatmap(
                x=axis, y=axis, z=grid, zmin=0, zmax=2,
                colorscale=[[0, "#f5f7fa"], [.49, "#f5f7fa"],
                            [.5, "#2664b4"], [.99, "#2664b4"], [1, "#191919"]],
                colorbar={"tickvals": [0, 1, 2],
                          "ticktext": ["自由", "内部", "边界"]},
                hovertemplate="x=%{x:.6f}<br>y=%{y:.6f}<br>显示分类=%{z}<extra></extra>"))
            if private:
                drivers = scene.driver_position.as_array()
                near = np.abs(drivers[:, 2] - z) <= 1 / max(resolution - 1, 1)
                fig.add_trace(go.Scatter(x=drivers[near, 0], y=drivers[near, 1],
                                         mode="markers", name="邻近切片的私有驱动",
                                         marker={"size": 10, "symbol": "diamond",
                                                 "color": "#e8a422"}))
            fig.update_layout(height=560, xaxis={"range": [0, 1], "scaleanchor": "y"},
                              yaxis={"range": [0, 1]},
                              margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(fig, width="stretch")
            st.caption("格点之间的边界形状只是显示插值，不能作为精确几何证据。")

    with right:
        st.subheader("精确查询")
        point = _coordinates("查询点", (.5, .5, z), "point")
        membership = view.classify_points(point)
        names = {Membership.FREE: "FREE", Membership.INTERIOR: "INTERIOR",
                 Membership.BOUNDARY: "BOUNDARY", Membership.UNKNOWN: "UNKNOWN"}
        st.metric("结构分类", names.get(int(membership.value[0]), "UNKNOWN"))
        st.json(_result(membership))
        radius = st.number_input("净空半径", min_value=0.0, value=.05,
                                 format="%.17g")
        clearance = view.clearance_decide(point, radius)
        st.json({"clearance": _result(clearance),
                 "witness_verified": view.verify_clearance_witness(
                     point, radius, clearance.evidence_ref[0])
                 if clearance.evidence_ref[0] is not None else None})
        if private:
            response = (PrivateOracle(scene, view).query_response(point)
                        if scene.physics.codec_id == 0 else
                        AdditiveOracle(scene, view).query_response(point))
            st.json({"private_response": _result(response),
                     "driver_positions": scene.driver_position.as_array().tolist()})
        with st.expander("线段与盒体"):
            a = _coordinates("线段起点", (0, .5, z), "segment-a")
            b = _coordinates("线段终点", (1, .5, z), "segment-b")
            segment = view.segment_intersect(a, b)
            st.json({"segment": _result(segment)})
            low = _coordinates("盒体下界", (0, 0, 0), "box-low")
            high = _coordinates("盒体上界", (1, 1, 1), "box-high")
            if any(low[i] > high[i] for i in range(3)):
                st.warning("盒体下界必须逐坐标不大于上界。")
            else:
                st.json({"box": _result(view.classify_box([low, high]))})
        payload = {"snapshot_id": snapshot_id, "scene_id": scene.content_hash.hex(),
                   "geometry_id": scene.geometry_id.hex(), "point": point,
                   "membership": _result(membership), "clearance_radius": radius,
                   "clearance": _result(clearance)}
        st.download_button("下载当前查询 JSON", data=json.dumps(payload, indent=2),
                           file_name="rind-3d-query.json", mime="application/json")

    with st.expander("几何表达与候选记录"):
        geometry = next(item for item in bundle.geometries
                        if item.content_hash == scene.geometry_id)
        st.json({"capability": capability, "root_kind": geometry.root_kind.name,
                 "tree": None if geometry.root_kind == RootKind.EMPTY
                 else _tree(bundle, geometry.root_hash)})
        st.dataframe([row for row in ledger if row["slot_index"] == selected["slot_index"]])


if __name__ == "__main__":
    main()
