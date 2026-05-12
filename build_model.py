#!/usr/bin/env python3
"""Build the MuJoCo MJCF for the DDSM115 4WD-A robot from meshes/assembly.json.

Reads ``meshes/assembly.json``, slugs each Chinese/CJK component name to an
ASCII role (wheel_fr, frame_extrusion_1, top_plate, ...), copies each STL into
``assets/<role>.stl``, and writes ``ddsm115_4wd.xml``.

MJCF conventions (from the spec):
  * compiler angle=radian, autolimits=true, meshes scale 0.001 (mm -> m)
  * chassis frame: +Y forward, +X right, +Z up
  * base_link freejoint body at world (0, 0, wheel_radius+0.001)
  * non-wheel components: fixed mesh geoms attached to base_link with pos/quat
    from each component's 4x4 placement
  * wheels: child bodies at component placement, hinge joint axis="0 1 0"
    (wheel-local Y == real axle), visual mesh + a thin cylinder collider
    rotated by quat "0.7071068 0.7071068 0 0" so its axis aligns with body-Y
  * default classes: chassis (contype=1 density=500), wheel_visual (contype=0
    density=0), wheel_collision (contype=1 density=800 friction=1.2 0.01 0.001)
  * 4 motor actuators ctrlrange="-3 3", gear=-1 on the two left motors so
    ctrl=+1 on all four drives forward
  * sensors: jointvel per wheel + framepos/framequat on base_link
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import trimesh


# --- raw_name -> ASCII role ----------------------------------------------

# Wheels are identified by chassis-frame position (sign of X, Y).
def wheel_role(tx_mm: float, ty_mm: float) -> str:
    side = "r" if tx_mm > 0 else "l"
    fb = "f" if ty_mm > 0 else "b"
    return f"wheel_{fb}{side}"


def assign_roles(records: list[dict]) -> list[dict]:
    """Walk components, attach a stable ASCII ``role`` string to each."""
    bracket_idx = extrusion_idx = side_idx = 0
    for rec in records:
        proto = rec["proto_name"]
        tx, ty, _ = rec["transform_mm"][0][3], rec["transform_mm"][1][3], rec["transform_mm"][2][3]
        if proto.startswith("DDSM115"):
            rec["role"] = wheel_role(tx, ty)
            rec["kind"] = "wheel"
        elif proto.startswith("CNC"):
            bracket_idx += 1
            rec["role"] = f"cnc_link_{bracket_idx}"
            rec["kind"] = "chassis"
        elif proto.startswith("2040"):
            extrusion_idx += 1
            rec["role"] = f"frame_extrusion_{extrusion_idx}"
            rec["kind"] = "chassis"
        elif proto.startswith("侧"):  # 侧 (side)
            side_idx += 1
            rec["role"] = f"side_support_{side_idx}"
            rec["kind"] = "chassis"
        elif proto.startswith("盖"):  # 盖 (cover/top)
            rec["role"] = "top_plate"
            rec["kind"] = "chassis"
        elif proto.startswith("零"):  # 零 (part)
            rec["role"] = "bottom_plate"
            rec["kind"] = "chassis"
        else:
            raise SystemExit(f"unknown proto: {proto!r}")
    return records


# --- math helpers --------------------------------------------------------

def quat_from_matrix(M: np.ndarray) -> tuple[float, float, float, float]:
    """Return (w, x, y, z) MuJoCo-style quaternion from 3x3 rotation matrix."""
    R = M[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def fmt3(v) -> str:
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def fmt4(v) -> str:
    return f"{v[0]:.7f} {v[1]:.7f} {v[2]:.7f} {v[3]:.7f}"


# --- wheel geometry from the actual mesh ---------------------------------

def measure_wheel(mesh_path: Path) -> tuple[float, float, float]:
    """Compute (wheel_radius_m, wheel_half_width_m, tire_center_y_m).

    Wheel-local Y is the axle. Radius = max sqrt(x^2 + z^2) of any vertex.
    Tire band = vertices within 5% of that max radius; half-width is half of
    that band's Y extent, centered at its midpoint along Y.
    """
    m = trimesh.load_mesh(mesh_path)
    v = m.vertices  # mm
    r = np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2)
    r_max = float(r.max())
    band = v[r > 0.95 * r_max]
    y_min, y_max = float(band[:, 1].min()), float(band[:, 1].max())
    return (r_max * 1e-3, (y_max - y_min) / 2 * 1e-3, (y_max + y_min) / 2 * 1e-3)


# --- MJCF emission -------------------------------------------------------

def build(meshes_dir: str, out_xml: str, assets_dir: str) -> None:
    meshes = Path(meshes_dir)
    assets = Path(assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    sidecar = json.loads((meshes / "assembly.json").read_text(encoding="utf-8"))
    records = assign_roles(sidecar["components"])

    # Copy each component STL to assets/<role>.stl.
    for rec in records:
        src = meshes / rec["stl"]
        dst = assets / f"{rec['role']}.stl"
        shutil.copyfile(src, dst)
        rec["asset_file"] = dst.name

    # Measure wheel dims from one wheel STL (all four are equivalent).
    wheel_rec = next(r for r in records if r["kind"] == "wheel")
    wheel_r, wheel_hw, wheel_y_center = measure_wheel(assets / f"{wheel_rec['role']}.stl")
    print(f"wheel_radius={wheel_r:.5f} m, wheel_half_width={wheel_hw:.5f} m, "
          f"tire_y_center={wheel_y_center:.5f} m")

    base_z = wheel_r + 0.001

    # Build the XML.
    mujoco = ET.Element("mujoco", {"model": "ddsm115_4wd"})

    ET.SubElement(mujoco, "compiler", {
        "angle": "radian",
        "autolimits": "true",
        "meshdir": "assets",
    })
    ET.SubElement(mujoco, "option", {
        "timestep": "0.002",
        "integrator": "implicitfast",
    })

    default = ET.SubElement(mujoco, "default")
    chassis = ET.SubElement(default, "default", {"class": "chassis"})
    ET.SubElement(chassis, "geom", {
        "type": "mesh", "contype": "1", "conaffinity": "1",
        "density": "500", "rgba": "0.55 0.55 0.6 1",
    })
    wv = ET.SubElement(default, "default", {"class": "wheel_visual"})
    ET.SubElement(wv, "geom", {
        "type": "mesh", "contype": "0", "conaffinity": "0",
        "density": "0", "rgba": "0.15 0.15 0.15 1",
    })
    wc = ET.SubElement(default, "default", {"class": "wheel_collision"})
    ET.SubElement(wc, "geom", {
        "type": "cylinder", "contype": "1", "conaffinity": "1",
        "density": "800", "friction": "1.2 0.01 0.001",
        "rgba": "0.1 0.1 0.1 0.3",
    })

    # Asset table: one mesh per role.
    asset = ET.SubElement(mujoco, "asset")
    for rec in records:
        ET.SubElement(asset, "mesh", {
            "name": rec["role"],
            "file": rec["asset_file"],
            "scale": "0.001 0.001 0.001",
        })

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 2", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "5 5 0.1",
        "rgba": "0.8 0.8 0.8 1", "contype": "1", "conaffinity": "1",
    })

    base = ET.SubElement(worldbody, "body", {
        "name": "base_link", "pos": f"0 0 {base_z:.6f}",
    })
    ET.SubElement(base, "freejoint", {"name": "root"})
    ET.SubElement(base, "site", {"name": "imu", "pos": "0 0 0", "size": "0.005"})

    # Non-wheel chassis components: fixed mesh geoms.
    for rec in records:
        if rec["kind"] != "chassis":
            continue
        M = np.array(rec["transform_mm"])
        pos = M[:3, 3] * 1e-3
        quat = quat_from_matrix(M)
        ET.SubElement(base, "geom", {
            "name": rec["role"],
            "class": "chassis",
            "mesh": rec["role"],
            "pos": fmt3(pos),
            "quat": fmt4(quat),
        })

    # Wheels: one body per wheel, with hinge joint, mesh visual, cylinder
    # collider rotated to align with body-local Y.
    wheel_quat_collider = "0.7071068 0.7071068 0 0"
    wheel_recs = [r for r in records if r["kind"] == "wheel"]
    for rec in wheel_recs:
        M = np.array(rec["transform_mm"])
        pos = M[:3, 3] * 1e-3
        quat = quat_from_matrix(M)
        wb = ET.SubElement(base, "body", {
            "name": rec["role"],
            "pos": fmt3(pos),
            "quat": fmt4(quat),
        })
        ET.SubElement(wb, "joint", {
            "name": f"{rec['role']}_hinge",
            "type": "hinge",
            "axis": "0 1 0",
            "damping": "0.05",
            "armature": "0.01",
        })
        ET.SubElement(wb, "geom", {
            "name": f"{rec['role']}_visual",
            "class": "wheel_visual",
            "mesh": rec["role"],
        })
        ET.SubElement(wb, "geom", {
            "name": f"{rec['role']}_col",
            "class": "wheel_collision",
            "size": f"{wheel_r:.6f} {wheel_hw:.6f}",
            "pos": f"0 {wheel_y_center:.6f} 0",
            "quat": wheel_quat_collider,
        })

    # Actuators: 4 motors, gear=-1 on the two left wheels (X<0).
    act = ET.SubElement(mujoco, "actuator")
    for rec in wheel_recs:
        tx = rec["transform_mm"][0][3]
        gear = "-1" if tx < 0 else "1"
        ET.SubElement(act, "motor", {
            "name": f"motor_{rec['role']}",
            "joint": f"{rec['role']}_hinge",
            "ctrlrange": "-3 3",
            "gear": gear,
        })

    # Sensors.
    sensor = ET.SubElement(mujoco, "sensor")
    for rec in wheel_recs:
        ET.SubElement(sensor, "jointvel", {
            "name": f"vel_{rec['role']}",
            "joint": f"{rec['role']}_hinge",
        })
    ET.SubElement(sensor, "framepos", {
        "name": "base_pos", "objtype": "body", "objname": "base_link",
    })
    ET.SubElement(sensor, "framequat", {
        "name": "base_quat", "objtype": "body", "objname": "base_link",
    })

    # Pretty-print.
    raw = ET.tostring(mujoco, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    Path(out_xml).write_text(pretty, encoding="utf-8")
    print(f"Wrote {out_xml}")


if __name__ == "__main__":
    meshes = sys.argv[1] if len(sys.argv) > 1 else "meshes"
    out_xml = sys.argv[2] if len(sys.argv) > 2 else "ddsm115_4wd.xml"
    assets = sys.argv[3] if len(sys.argv) > 3 else "assets"
    build(meshes, out_xml, assets)
