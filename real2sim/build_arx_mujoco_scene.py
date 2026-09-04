#!/usr/bin/env python3
"""Build a MuJoCo MJCF scene for the ARX AC one robot."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from assets import DEFAULT_ASSETS, MUJOCO_ARX_SCENE

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARX_ROOT = Path(DEFAULT_ASSETS.platform.urdf_root)
DEFAULT_OUT_DIR = MUJOCO_ARX_SCENE.parent

DEFAULT_TABLE_TOP_Z_M = float(DEFAULT_ASSETS.platform.table_top_z_m)
DEFAULT_FLOOR_Z_M = float(DEFAULT_ASSETS.platform.floor_z_m)
DEFAULT_TABLE_CENTER_M = DEFAULT_ASSETS.platform.table_center_m
DEFAULT_TABLE_HALF_SIZE_M = DEFAULT_ASSETS.platform.table_half_size_m
DEFAULT_WORKSPACE_CENTER_M = DEFAULT_ASSETS.platform.workspace_center_m
DEFAULT_WORKSPACE_HALF_SIZE_M = DEFAULT_ASSETS.platform.workspace_half_size_m
DEFAULT_HEAD_CAMERA = DEFAULT_ASSETS.camera()

GRIPPER_JOINTS = {"left_joint7", "left_joint8", "right_joint17", "right_joint18"}
LEFT_TCP_OFFSET_M = DEFAULT_ASSETS.platform.left_tcp.t_flange_m
RIGHT_TCP_OFFSET_M = DEFAULT_ASSETS.platform.right_tcp.t_flange_m

LINK_COLORS = {
    "base_link": "0.18 0.18 0.20 1",
    "left_link1": "0.82 0.84 0.90 1",
    "left_link2": "0.70 0.73 0.78 1",
    "left_link3": "0.88 0.86 0.82 1",
    "left_link4": "0.50 0.50 0.52 1",
    "left_link5": "0.62 0.63 0.64 1",
    "left_link6": "0.86 0.88 0.90 1",
    "left_link7": "0.94 0.94 0.92 1",
    "left_link8": "0.94 0.94 0.92 1",
    "right_link11": "0.82 0.84 0.90 1",
    "right_link12": "0.70 0.73 0.78 1",
    "right_link13": "0.88 0.86 0.82 1",
    "right_link14": "0.70 0.73 0.78 1",
    "right_link15": "0.62 0.63 0.64 1",
    "right_link16": "0.86 0.88 0.90 1",
    "right_link17": "0.94 0.94 0.92 1",
    "right_link18": "0.94 0.94 0.92 1",
}


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_arx_root(cli_root: str | None = None) -> Path:
    candidates = [
        cli_root,
        os.environ.get("ARX_ACONE_URDF_ROOT"),
        DEFAULT_ARX_ROOT,
    ]
    tried = []
    for candidate in candidates:
        if not candidate:
            continue
        root = as_abs(candidate)
        tried.append(str(root))
        if find_urdf_path(root) is not None and (root / "meshes").is_dir():
            return root
    raise FileNotFoundError(
        "Could not find ARX AC one URDF root. Use --arx-root or ARX_ACONE_URDF_ROOT. "
        "Tried: " + ", ".join(tried)
    )


def find_urdf_path(arx_root: Path) -> Path | None:
    for filename in ("ACone.urdf", "acone.urdf"):
        path = arx_root / "urdf" / filename
        if path.is_file():
            return path
    matches = sorted((arx_root / "urdf").glob("*.urdf")) if (arx_root / "urdf").is_dir() else []
    return matches[0] if matches else None


def parse_vec(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(v) for v in (text or default).split()]


def fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def np_list(values) -> list:
    return np_list(values.tolist()) if hasattr(values, "tolist") else values


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def vec_mul(scalar: float, vec: list[float]) -> list[float]:
    return [scalar * vec[i] for i in range(3)]


def rpy_to_quat(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ]


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def mesh_name_from_link(link: str) -> str:
    return f"{link}_mesh"


def urdf_tag(node: ET.Element) -> str:
    return node.tag.split("}", 1)[-1]


def mesh_filename(link_node: ET.Element) -> str | None:
    visual = link_node.find("visual")
    if visual is None:
        return None
    mesh = visual.find("./geometry/mesh")
    if mesh is None:
        return None
    filename = mesh.attrib.get("filename")
    if not filename:
        return None
    return Path(filename).name


def visual_origin(link_node: ET.Element) -> tuple[list[float], list[float]]:
    visual = link_node.find("visual")
    if visual is None:
        return parse_vec(None), rpy_to_quat(parse_vec(None))
    origin = visual.find("origin")
    return (
        parse_vec(origin.attrib.get("xyz") if origin is not None else None),
        rpy_to_quat(parse_vec(origin.attrib.get("rpy") if origin is not None else None)),
    )


def load_urdf_model(arx_root: Path) -> tuple[dict[str, ET.Element], dict[str, ET.Element], dict[str, list[dict]]]:
    urdf_path = find_urdf_path(arx_root)
    if urdf_path is None:
        raise FileNotFoundError(f"Could not find an AC one URDF file under {arx_root / 'urdf'}")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: dict[str, ET.Element] = {}
    joints: dict[str, ET.Element] = {}
    for node in list(root):
        tag = urdf_tag(node)
        if tag == "link":
            links[node.attrib["name"]] = node
        elif tag == "joint":
            joints[node.attrib["name"]] = node

    children: dict[str, list[dict]] = {}
    for name, joint in joints.items():
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append({"name": name, "joint": joint, "child": child})
    return links, joints, children


def copy_meshes(out_dir: Path, arx_root: Path, links: dict[str, ET.Element]) -> None:
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for link, link_node in links.items():
        filename = mesh_filename(link_node)
        if not filename:
            continue
        src = arx_root / "meshes" / filename
        if not src.is_file():
            raise FileNotFoundError(f"Missing mesh for {link}: {src}")
        shutil.copy2(src, mesh_dir / filename)


def add_inertial(body: ET.Element, link_node: ET.Element) -> None:
    inertial = link_node.find("inertial")
    if inertial is None:
        return
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        return
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": fmt(parse_vec(origin.attrib.get("xyz") if origin is not None else None)),
            "mass": mass.attrib["value"],
            "fullinertia": fmt(
                [
                    inertia.attrib["ixx"],
                    inertia.attrib["iyy"],
                    inertia.attrib["izz"],
                    inertia.attrib["ixy"],
                    inertia.attrib["ixz"],
                    inertia.attrib["iyz"],
                ]
            ),
        },
    )


def add_link_geom(body: ET.Element, link: str, link_node: ET.Element) -> None:
    filename = mesh_filename(link_node)
    if not filename:
        return
    pos, quat = visual_origin(link_node)
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{link}_visual",
            "type": "mesh",
            "mesh": mesh_name_from_link(link),
            "pos": fmt(pos),
            "quat": fmt(quat),
            "rgba": LINK_COLORS.get(link, "0.75 0.75 0.78 1"),
            "contype": "0",
            "conaffinity": "0",
        },
    )


def joint_limit(joint: ET.Element) -> tuple[float, float]:
    limit = joint.find("limit")
    if limit is None:
        return -math.pi, math.pi
    return float(limit.attrib.get("lower", -math.pi)), float(limit.attrib.get("upper", math.pi))


def add_sites(body: ET.Element, link: str) -> None:
    if link == "left_link6":
        ET.SubElement(body, "site", {"name": "left_flange", "pos": "0 0 0", "size": "0.018", "rgba": "0.15 0.95 0.25 1"})
        ET.SubElement(body, "site", {"name": "left_tcp", "pos": fmt(LEFT_TCP_OFFSET_M), "size": "0.020", "rgba": "0.95 0.15 0.85 1"})
        ET.SubElement(body, "site", {"name": "tcp", "pos": fmt(LEFT_TCP_OFFSET_M), "size": "0.022", "rgba": "1 0.25 0.95 1"})
    elif link == "right_link16":
        ET.SubElement(body, "site", {"name": "right_flange", "pos": "0 0 0", "size": "0.018", "rgba": "0.15 0.95 0.25 1"})
        ET.SubElement(body, "site", {"name": "right_tcp", "pos": fmt(RIGHT_TCP_OFFSET_M), "size": "0.020", "rgba": "0.95 0.15 0.85 1"})


def add_body_recursive(
    parent_body: ET.Element,
    link: str,
    links: dict[str, ET.Element],
    children: dict[str, list[dict]],
    joint_order: list[str],
) -> None:
    add_inertial(parent_body, links[link])
    add_link_geom(parent_body, link, links[link])
    add_sites(parent_body, link)

    for entry in children.get(link, []):
        joint = entry["joint"]
        child = entry["child"]
        origin = joint.find("origin")
        body = ET.SubElement(
            parent_body,
            "body",
            {
                "name": child,
                "pos": fmt(parse_vec(origin.attrib.get("xyz") if origin is not None else None)),
                "quat": fmt(rpy_to_quat(parse_vec(origin.attrib.get("rpy") if origin is not None else None))),
            },
        )
        joint_type = joint.attrib["type"]
        if joint_type in {"revolute", "continuous"}:
            axis = parse_vec(joint.find("axis").attrib.get("xyz", "0 0 1"))
            lower, upper = joint_limit(joint)
            attrs = {
                "name": entry["name"],
                "type": "hinge",
                "axis": fmt(axis),
                "damping": "1.0",
                "armature": "0.02",
            }
            if joint_type == "revolute":
                attrs.update({"limited": "true", "range": f"{lower:.9g} {upper:.9g}"})
            ET.SubElement(body, "joint", attrs)
            joint_order.append(entry["name"])
        elif joint_type == "prismatic":
            axis = parse_vec(joint.find("axis").attrib.get("xyz", "0 0 1"))
            lower, upper = joint_limit(joint)
            ET.SubElement(
                body,
                "joint",
                {
                    "name": entry["name"],
                    "type": "slide",
                    "axis": fmt(axis),
                    "limited": "true",
                    "range": f"{lower:.9g} {upper:.9g}",
                    "damping": "2.0",
                    "armature": "0.002",
                },
            )
            joint_order.append(entry["name"])
        elif joint_type != "fixed":
            raise ValueError(f"Unsupported URDF joint type {joint_type!r} for {entry['name']}")
        add_body_recursive(body, child, links, children, joint_order)


def add_axis(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "geom", {"name": "base_plus_x_forward_axis", "type": "capsule", "fromto": "0 0 0.025 0.55 0 0.025", "size": "0.008", "rgba": "0.9 0.1 0.1 1"})
    ET.SubElement(worldbody, "geom", {"name": "base_minus_x_axis", "type": "capsule", "fromto": "0 0 0.025 -0.35 0 0.025", "size": "0.005", "rgba": "0.55 0.05 0.05 1"})
    ET.SubElement(worldbody, "geom", {"name": "base_plus_y_axis", "type": "capsule", "fromto": "0 0 0.035 0 0.45 0.035", "size": "0.008", "rgba": "0.1 0.75 0.15 1"})
    ET.SubElement(worldbody, "geom", {"name": "base_minus_y_axis", "type": "capsule", "fromto": "0 0 0.035 0 -0.45 0.035", "size": "0.005", "rgba": "0.05 0.45 0.08 1"})
    workspace_site_pos = [0.40, 0.0, DEFAULT_TABLE_TOP_Z_M + 0.025]
    ET.SubElement(worldbody, "site", {"name": "robot_front_workspace_xpos", "pos": fmt(workspace_site_pos), "size": "0.025", "rgba": "1 0.8 0.05 1"})


def add_camera_setup(worldbody: ET.Element) -> None:
    if DEFAULT_HEAD_CAMERA is None or not DEFAULT_HEAD_CAMERA.extrinsics.is_filled():
        raise ValueError("DEFAULT_ASSETS.camera().extrinsics.T_cam_in_base must be filled")
    intr = DEFAULT_HEAD_CAMERA.intrinsics
    T_cam_in_base = DEFAULT_HEAD_CAMERA.extrinsics.T_cam_in_base
    camera_pos = [T_cam_in_base[i][3] for i in range(3)]
    x_axis = [T_cam_in_base[i][0] for i in range(3)]
    y_axis = [-T_cam_in_base[i][1] for i in range(3)]
    z_forward = [T_cam_in_base[i][2] for i in range(3)]
    table_top_z = float(DEFAULT_TABLE_TOP_Z_M)
    if abs(z_forward[2]) > 1e-9:
        target = vec_add(camera_pos, vec_mul((table_top_z - camera_pos[2]) / z_forward[2], z_forward))
    else:
        target = vec_add(camera_pos, vec_mul(0.65, z_forward))
    if intr.height is not None and intr.fy is not None:
        fovy = math.degrees(2.0 * math.atan((intr.height * 0.5) / intr.fy))
    else:
        fovy = 70.0

    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "ego_camera_calibrated",
            "pos": fmt(camera_pos),
            "xyaxes": fmt(list(x_axis) + list(y_axis)),
            "fovy": f"{fovy:.3f}",
        },
    )
    camera_body = ET.SubElement(worldbody, "body", {"name": "camera_body", "pos": fmt(camera_pos)})
    ET.SubElement(camera_body, "geom", {"name": "camera_box", "type": "box", "size": "0.012 0.020 0.014", "rgba": "0.05 0.08 0.10 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(worldbody, "geom", {"name": "camera_optical_axis", "type": "capsule", "fromto": f"{fmt(camera_pos)} {fmt(target)}", "size": "0.004", "rgba": "0.1 0.35 1 0.75"})
    ET.SubElement(worldbody, "site", {"name": "camera_forward_target", "pos": fmt(target), "size": "0.018", "rgba": "0.1 0.35 1 1"})


def add_humanego_eef_marker(worldbody: ET.Element) -> None:
    marker = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "humanego_eef_marker",
            "mocap": "true",
            "pos": "0 0 0.2",
            "quat": "1 0 0 0",
        },
    )
    ET.SubElement(marker, "geom", {"name": "humanego_eef_origin", "type": "sphere", "size": "0.018", "rgba": "0.05 0.75 1 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"name": "humanego_eef_x_axis", "type": "capsule", "fromto": "0 0 0 0.075 0 0", "size": "0.005", "rgba": "1 0.05 0.05 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"name": "humanego_eef_y_axis", "type": "capsule", "fromto": "0 0 0 0 0.075 0", "size": "0.005", "rgba": "0.05 0.8 0.1 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"name": "humanego_eef_z_axis", "type": "capsule", "fromto": "0 0 0 0 0 0.075", "size": "0.005", "rgba": "0.1 0.3 1 1", "contype": "0", "conaffinity": "0"})


def actuator_kp(joint_name: str) -> str:
    return "40" if joint_name in GRIPPER_JOINTS else "80"


def initial_qpos(joint_name: str, joints: dict[str, ET.Element]) -> float:
    lower, upper = joint_limit(joints[joint_name])
    if joint_name in GRIPPER_JOINTS:
        return upper
    return min(max(0.0, lower), upper)


def build_scene(out_dir: Path, arx_root: Path) -> Path:
    links, joints, children = load_urdf_model(arx_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_meshes(out_dir, arx_root, links)

    root = ET.Element("mujoco", {"model": "arx_acone_table_camera_scene"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "meshes", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    ET.SubElement(root, "statistic", {"center": "0.22 0 0.22", "extent": "1.2"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.65 0.65 0.65", "specular": "0.15 0.15 0.15"})
    ET.SubElement(visual, "rgba", {"haze": "0.30 0.45 0.65 1"})
    ET.SubElement(visual, "global", {"azimuth": "145", "elevation": "-25", "offwidth": "1280", "offheight": "720"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "skybox", "type": "skybox", "builtin": "gradient", "rgb1": "0.30 0.50 0.70", "rgb2": "0 0 0", "width": "32", "height": "512"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.20 0.30 0.40", "rgb2": "0.10 0.15 0.20", "width": "512", "height": "512", "mark": "cross", "markrgb": "0.8 0.8 0.8"})
    ET.SubElement(asset, "material", {"name": "floor_mat", "texture": "grid", "texrepeat": "2 2", "texuniform": "true", "reflectance": "0.2"})
    ET.SubElement(asset, "material", {"name": "table_mat", "rgba": "0.62 0.62 0.58 1"})
    ET.SubElement(asset, "material", {"name": "table_leg_mat", "rgba": "0.30 0.30 0.32 1"})
    ET.SubElement(asset, "material", {"name": "workspace_mat", "rgba": "1 0.82 0.12 0.25"})
    for link, link_node in links.items():
        filename = mesh_filename(link_node)
        if filename:
            ET.SubElement(asset, "mesh", {"name": mesh_name_from_link(link), "file": filename})

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "0.7 -0.8 1.2", "dir": "-0.5 0.4 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "pos": f"0 0 {DEFAULT_FLOOR_Z_M:.9g}", "size": "0 0 0.05", "material": "floor_mat"})
    ET.SubElement(worldbody, "geom", {"name": "table_top", "type": "box", "pos": fmt(DEFAULT_TABLE_CENTER_M), "size": fmt(DEFAULT_TABLE_HALF_SIZE_M), "material": "table_mat"})
    leg_half_height = 0.5 * (DEFAULT_TABLE_CENTER_M[2] - DEFAULT_TABLE_HALF_SIZE_M[2] - DEFAULT_FLOOR_Z_M)
    leg_center_z = DEFAULT_FLOOR_Z_M + leg_half_height
    leg_i = 0
    for x in (
        DEFAULT_TABLE_CENTER_M[0] - DEFAULT_TABLE_HALF_SIZE_M[0] + 0.08,
        DEFAULT_TABLE_CENTER_M[0] + DEFAULT_TABLE_HALF_SIZE_M[0] - 0.08,
    ):
        for y in (
            DEFAULT_TABLE_CENTER_M[1] - DEFAULT_TABLE_HALF_SIZE_M[1] + 0.08,
            DEFAULT_TABLE_CENTER_M[1] + DEFAULT_TABLE_HALF_SIZE_M[1] - 0.08,
        ):
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"table_leg_{leg_i}",
                    "type": "box",
                    "pos": fmt([x, y, leg_center_z]),
                    "size": fmt([0.025, 0.025, leg_half_height]),
                    "material": "table_leg_mat",
                },
            )
            leg_i += 1
    workspace_pos = DEFAULT_WORKSPACE_CENTER_M.copy()
    workspace_pos[2] = DEFAULT_TABLE_TOP_Z_M + DEFAULT_WORKSPACE_HALF_SIZE_M[2]
    ET.SubElement(worldbody, "geom", {"name": "front_workspace_patch", "type": "box", "pos": fmt(workspace_pos), "size": fmt(DEFAULT_WORKSPACE_HALF_SIZE_M), "material": "workspace_mat", "contype": "0", "conaffinity": "0"})
    add_axis(worldbody)
    add_camera_setup(worldbody)

    joint_order: list[str] = []
    base_body = ET.SubElement(worldbody, "body", {"name": "base_link", "pos": "0 0 0", "quat": "1 0 0 0"})
    add_body_recursive(base_body, "base_link", links, children, joint_order)
    add_humanego_eef_marker(worldbody)

    actuator = ET.SubElement(root, "actuator")
    for joint_name in joint_order:
        lower, upper = joint_limit(joints[joint_name])
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_pos",
                "joint": joint_name,
                "kp": actuator_kp(joint_name),
                "ctrlrange": f"{lower:.9g} {upper:.9g}",
            },
        )

    qpos = [initial_qpos(joint_name, joints) for joint_name in joint_order]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "open", "qpos": fmt(qpos), "ctrl": fmt(qpos)})

    indent(root)
    scene_path = out_dir / "scene.xml"
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)
    return scene_path


def write_readme(out_dir: Path, arx_root: Path, scene_path: Path) -> None:
    camera = DEFAULT_ASSETS.camera()
    intr = camera.intrinsics
    extr = camera.extrinsics
    workspace_center = np_list(DEFAULT_WORKSPACE_CENTER_M)
    table_center = np_list(DEFAULT_TABLE_CENTER_M)
    table_half_size = np_list(DEFAULT_TABLE_HALF_SIZE_M)
    T_cam_in_base = np_list(extr.T_cam_in_base)
    bounds_min = np_list(extr.source_mesh_bounds_min_m)
    bounds_max = np_list(extr.source_mesh_bounds_max_m)
    readme = f"""# ARX AC one MuJoCo Scene

Generated from `{arx_root}`.

Files:
- `scene.xml`: MuJoCo MJCF scene.
- `meshes/*.STL`: local ARX mesh assets copied from the URDF package.

Scene convention:
- Robot base frame remains at `z=0`. The rectangular base bottom sits on the tabletop at `z={DEFAULT_TABLE_TOP_Z_M}`.
- The full AC one URDF is loaded: left arm, right arm, and both two-finger prismatic grippers.
- ARX working direction is base `+X`; `+Z` is up.
- The normal workspace marker is centered in front of the robot at `{workspace_center}` m.
- The tabletop is a box centered at `{table_center}` m with half-size `{table_half_size}` m.
- The scene uses a MuJoCo-style blue checker floor plane at `z={DEFAULT_FLOOR_Z_M}`, 0.70 m below the tabletop.
- The HEAD camera pose is estimated from the middle head/camera component on `base_link.STL`.
- HEAD camera source mesh bounds are min `{bounds_min}` m, max `{bounds_max}` m.

HEAD camera:
- Device: `{camera.capture.get("device", "unknown")}`.
- Resolution: `{intr.width}x{intr.height}`.
- Distortion model: `{intr.distortion_model}`.
- K:
  - `[{intr.fx}, 0, {intr.cx}]`
  - `[0, {intr.fy}, {intr.cy}]`
  - `[0, 0, 1]`
- Distortion coeffs `[k1, k2, p1, p2, k3]`: `{np_list(intr.dist_coeffs)}`.
- `T_cam_in_base` (OpenCV camera axes as columns): `{T_cam_in_base}`.
- Optical axis points toward base `+X` and downward by `{extr.pitch_down_deg}` deg.

Coordinate note:
- MuJoCo is right-handed. With `+X` forward and `+Z` up, the URDF's `left_*` branch is located at
  positive `Y` and the `right_*` branch is located at negative `Y`. The names come from the vendor
  URDF; downstream code should rely on frame names and measured positions rather than assuming
  the semantic side from the sign of `Y`.

Important frames:
- `left_flange`: left wrist body `left_link6` origin.
- `left_tcp`: approximate midpoint between the left gripper fingers; zero pose points along base `+X`.
- `tcp`: alias of `left_tcp` for single-arm pipeline compatibility.
- `right_flange`: right wrist body `right_link16` origin.
- `right_tcp`: approximate midpoint between the right gripper fingers; zero pose points along base `+X`.
- `humanego_eef_marker`: mocap target marker used by replay/retargeting scripts.

Open gripper keyframe:
- `open`: all arm revolute joints are clamped around zero, and all four prismatic finger joints are at their upper limits.

Load with MuJoCo:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("{scene_path}")
data = mujoco.MjData(model)
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arx-root", default=None, help="Path to the AC one URDF package root.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory for scene.xml and meshes.")
    args = parser.parse_args()

    arx_root = resolve_arx_root(args.arx_root)
    out_dir = as_abs(args.out)
    scene_path = build_scene(out_dir=out_dir, arx_root=arx_root)
    write_readme(out_dir=out_dir, arx_root=arx_root, scene_path=scene_path)
    print(f"Wrote {scene_path}")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
