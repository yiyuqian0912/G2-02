"""Local Streamlit inspector for committed RIND 2D truth snapshots.

Launch with: streamlit run dataset/src/rind/inspector.py -- --dataset PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

from rind.contracts import CSGNode, CSGOpcode, Geometry, RootKind, make_leaf
from rind.geometry import GeometryView, Membership
from rind.oracle import PrivateOracle
from rind.decay import AdditiveOracle
from rind.publication import validate_snapshot
from rind.storage import CanonicalBundle, read_bundle


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="dataset/artifacts/pilot-2d")
    return parser.parse_known_args()[0]


@st.cache_resource(show_spinner="验证并载入快照…")
def _load_snapshot(root: str, snapshot_id: str):
    summary = validate_snapshot(root)
    if summary["snapshot_id"] != snapshot_id:
        raise ValueError("CURRENT changed during load; refresh to pin a new snapshot")
    base = Path(root) / "snapshots" / snapshot_id
    bundle = read_bundle(base / "d2" / "core.zarr", 2)
    ledger = pq.read_table(base / "catalogs" / "candidate_ledger.parquet").to_pylist()
    return summary, bundle, ledger


def _box(bundle: CanonicalBundle, x0: float, y0: float,
         x1: float, y1: float) -> bytes:
    leaf = make_leaf("affine_box_v1", 2,
                     center=[(x0 + x1) / 2, (y0 + y1) / 2],
                     matrix=[[(x1 - x0) / 2, 0], [0, (y1 - y0) / 2]])
    bundle.add_leaf(leaf)
    return leaf.content_hash


@st.cache_resource(show_spinner=False)
def _fixture(name: str):
    bundle = CanonicalBundle(2)
    if name == "空结构":
        geometry = Geometry.empty(2)
        point = (.5, .5)
    elif name == "圆盘切触":
        leaf = make_leaf("affine_ball_v1", 2,
                         center=[.5, .5], matrix=[[.25, 0], [0, .25]])
        bundle.add_leaf(leaf)
        geometry = Geometry(2, RootKind.REF, leaf.content_hash)
        point = (.5, .75)
    elif name == "差集空腔":
        outer = _box(bundle, .125, .125, .875, .875)
        inner = _box(bundle, .375, .375, .625, .625)
        node = CSGNode(2, CSGOpcode.DIFFERENCE, (outer, inner))
        bundle.add_node(node)
        geometry = Geometry(2, RootKind.REF, node.content_hash)
        point = (.375, .5)
    elif name == "极窄自由缝":
        gap = 2**-40
        left = _box(bundle, .25, .25, .5, .75)
        right = _box(bundle, .5 + gap, .25, .75, .75)
        node = CSGNode(2, CSGOpcode.UNION, (left, right))
        bundle.add_node(node)
        geometry = Geometry(2, RootKind.REF, node.content_hash)
        point = (.5 + gap / 2, .5)
    else:
        raise ValueError(name)
    bundle.add_geometry(geometry)
    return bundle, geometry.content_hash, point


@st.cache_data(show_spinner="生成显示采样网格…")
def _grid(geometry_id: bytes, source_key: str, resolution: int,
          _bundle: CanonicalBundle) -> np.ndarray:
    view = GeometryView(geometry_id, _bundle)
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    result = view.classify_points(points)
    if not np.all(result.valid):
        raise RuntimeError("display grid contains unproved membership")
    return result.value.reshape(resolution, resolution)


def _reference_tree(bundle: CanonicalBundle, root_hash: bytes) -> dict:
    refs = {item.content_hash: item for item in bundle.references}
    item = refs[root_hash]
    if isinstance(item, CSGNode):
        return {"id": root_hash.hex(), "opcode": item.opcode.name,
                "children": [_reference_tree(bundle, child) for child in item.children]}
    return {"id": root_hash.hex(), "codec": item.codec.name}


def _describe_status(result, index: int = 0) -> dict:
    value = result.value[index]
    work = (result.work[index] if hasattr(result, "work") else
            {"precision_bits": int(result.precision_bits[index])})
    return {"status": int(result.status[index]),
            "value": float(value) if np.issubdtype(result.value.dtype, np.floating)
            else int(value),
            "valid": bool(result.valid[index]),
            "reason_code": result.reason_code[index], "work": work}


def main() -> None:
    st.set_page_config(page_title="RIND 结构真值查看器", layout="wide")
    st.title("RIND 连续结构真值查看器")
    st.caption("色块只是有限网格显示；下方点、线段、盒体与净空结果由精确几何后端判定。")
    source = st.sidebar.radio("数据源", ["数据集快照", "解析夹具"])
    scene = None
    snapshot_id = "fixture"
    if source == "数据集快照":
        root = st.sidebar.text_input("数据集目录", value=str(Path(_args().dataset).resolve()))
        try:
            snapshot_id = (Path(root) / "CURRENT").read_text().strip()
            summary, bundle, ledger = _load_snapshot(root, snapshot_id)
        except Exception as exc:
            st.error(f"快照无法载入：{exc}")
            st.stop()
        st.sidebar.success(f"已固定快照 {snapshot_id[:12]}")
        st.sidebar.metric("READY 候选", summary["ready_candidates"])
        st.sidebar.metric("唯一场景", summary["unique_ready_scenes"])
        ready = [row for row in ledger if row["candidate_state"] == "READY"]
        ready.sort(key=lambda row: row["slot_index"])
        chosen = st.sidebar.selectbox("候选槽位", ready,
                                      format_func=lambda row: f"槽位 {row['slot_index']:03d} · 变体 {row['scene_variant_index']} · {row['scene_id'][:10]}")
        scene = next(item for item in bundle.scenes if item.content_hash.hex() == chosen["scene_id"])
        geometry_id = scene.geometry_id
        initial_point = (.5, .5)
        st.info(f"候选 {chosen['candidate_id'][:16]} · 尝试 {chosen['attempt_index']} · 场景 {chosen['scene_id'][:16]}")
        st.caption(f"物理变体：{scene.physics.codec_id}（0 二值，1 固定指数，2 单调 log-PCHIP）")
    else:
        name = st.sidebar.selectbox("解析夹具", ["空结构", "圆盘切触", "差集空腔", "极窄自由缝"])
        bundle, geometry_id, initial_point = _fixture(name)
        if name == "极窄自由缝":
            st.info("自由缝宽度为 2⁻⁴⁰，显示网格通常无法分辨；精确点与线段查询仍能分辨。")
    private = st.sidebar.checkbox("显示私有驱动与响应", value=False,
                                  disabled=scene is None, key=f"private-{source}")
    resolution = st.sidebar.slider("显示采样分辨率", 32, 160, 80, step=16)
    view = GeometryView(geometry_id, bundle)
    if not view.capability()["supported"]:
        st.error(f"此几何后端不支持：{view.capability()['reason']}")
        st.stop()

    left, right = st.columns([1.5, 1])
    with left:
        grid = _grid(geometry_id, snapshot_id, resolution, bundle)
        axis = np.linspace(0, 1, resolution)
        figure = go.Figure(go.Heatmap(
            x=axis, y=axis, z=grid,
            zmin=0, zmax=2,
            colorscale=[[0, "#f5f7fa"], [.49, "#f5f7fa"],
                        [.5, "#2664b4"], [.99, "#2664b4"], [1, "#191919"]],
            colorbar={"tickvals": [0, 1, 2], "ticktext": ["自由", "内部", "边界"]},
            hovertemplate="x=%{x:.6f}<br>y=%{y:.6f}<br>显示分类=%{z}<extra></extra>",
        ))
        if scene is not None and private:
            drivers = scene.driver_position.as_array()
            figure.add_trace(go.Scatter(x=drivers[:, 0], y=drivers[:, 1],
                                        mode="markers", name="私有驱动",
                                        marker={"size": 10, "symbol": "diamond", "color": "#e8a422"}))
        figure.update_layout(height=590, xaxis={"range": [0, 1], "scaleanchor": "y"},
                             yaxis={"range": [0, 1]}, margin={"l": 20, "r": 20, "t": 20, "b": 20})
        st.plotly_chart(figure, width="stretch")
        st.caption("显示图可近似；任何采样边缘都不能替代精确边界。")

    with right:
        st.subheader("精确查询")
        x = st.number_input("x", 0.0, 1.0, float(initial_point[0]),
                            format="%.17g", key=f"qx-{geometry_id.hex()}")
        y = st.number_input("y", 0.0, 1.0, float(initial_point[1]),
                            format="%.17g", key=f"qy-{geometry_id.hex()}")
        point = [x, y]
        membership = view.classify_points(point)
        names = {Membership.FREE: "FREE", Membership.INTERIOR: "INTERIOR",
                 Membership.BOUNDARY: "BOUNDARY", Membership.UNKNOWN: "UNKNOWN"}
        st.metric("结构分类", names.get(int(membership.value[0]), "UNKNOWN"))
        st.json(_describe_status(membership))
        radius = st.number_input("净空判定半径", min_value=0.0, value=.05, format="%.17g")
        clearance = view.clearance_decide(point, radius)
        st.write("净空：", "LE" if clearance.value[0] == 0 else
                 "GT" if clearance.value[0] == 1 else "UNRESOLVED")
        if clearance.evidence_ref[0] is not None:
            witness = clearance.evidence_ref[0]
            st.caption(f"见证点显示坐标 ≈ ({witness.x_approx:.12g}, {witness.y_approx:.12g})；"
                       f"精确回代验证：{view.verify_clearance_witness(point, radius, witness)}")
        if private and scene is not None:
            response = (PrivateOracle(scene, view).query_response(point)
                        if scene.physics.codec_id == 0 else
                        AdditiveOracle(scene, view).query_response(point))
            st.write("私有响应：", float(response.value[0]) if response.valid[0] else "无效")
            st.json(_describe_status(response))
        with st.expander("线段与盒体"):
            ax = st.number_input("线段起点 x", 0.0, 1.0, 0.0, key="ax")
            ay = st.number_input("线段起点 y", 0.0, 1.0, .5, key="ay")
            bx = st.number_input("线段终点 x", 0.0, 1.0, 1.0, key="bx")
            by = st.number_input("线段终点 y", 0.0, 1.0, .5, key="by")
            st.json({"segment": _describe_status(view.segment_intersect([ax, ay], [bx, by]))})
            x0, y0 = st.columns(2)
            lo_x = x0.number_input("盒体左 x", 0.0, 1.0, 0.0)
            lo_y = y0.number_input("盒体下 y", 0.0, 1.0, 0.0)
            x1, y1 = st.columns(2)
            hi_x = x1.number_input("盒体右 x", 0.0, 1.0, 1.0)
            hi_y = y1.number_input("盒体上 y", 0.0, 1.0, 1.0)
            st.json({"box": _describe_status(view.classify_box([[lo_x, lo_y], [hi_x, hi_y]]))})
        export = {"geometry_id": geometry_id.hex(), "point": point,
                  "classification": _describe_status(membership),
                  "clearance_radius": radius,
                  "clearance": _describe_status(clearance),
                  "source": source, "snapshot_id": snapshot_id}
        st.download_button("下载当前查询 JSON", data=json.dumps(export, ensure_ascii=False, indent=2),
                           file_name="rind-query.json", mime="application/json")

    with st.expander("几何表达与能力声明"):
        geometry = next(item for item in bundle.geometries if item.content_hash == geometry_id)
        st.json(view.capability())
        st.json({"root_kind": geometry.root_kind.name,
                 "tree": None if geometry.root_kind == RootKind.EMPTY
                 else _reference_tree(bundle, geometry.root_hash)})


if __name__ == "__main__":
    main()
