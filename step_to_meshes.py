#!/usr/bin/env python3
"""Tessellate the DDSM115 4WD-A STEP assembly into per-component STLs.

Reads ``raw/DDSM115 4WD-A.step`` via STEPCAFControl_Reader, finds the root
assembly with ``GetFreeShapes``/``GetComponents``, tessellates each unique
prototype once with BRepMesh, and writes:

  meshes/comp_NN.stl       -- one mesh per component in the prototype-local
                              frame (millimeters)
  meshes/assembly.json     -- per-component metadata: stl filename, raw_name,
                              proto_name, 4x4 placement matrix in mm,
                              local-frame bbox in mm

Component placements are extracted from gp_Trsf via ``Value(r+1, c+1)``
(1-indexed). Face triangulations whose orientation is TopAbs_REVERSED get
their winding flipped so all STL faces point outward.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool


LINEAR_DEFLECTION = 0.3    # mm
ANGULAR_DEFLECTION = 0.5   # radians


def label_name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return "<unnamed>"


def trsf_to_matrix_mm(loc) -> np.ndarray:
    """Convert a TopLoc_Location/gp_Trsf into a 4x4 matrix (translation in mm)."""
    t = loc.Transformation()
    m = np.eye(4, dtype=float)
    for r in range(3):
        for c in range(4):
            m[r, c] = t.Value(r + 1, c + 1)
    return m


def tessellate_prototype(shape) -> trimesh.Trimesh:
    """Mesh every face of the given (possibly compound) shape into one Trimesh.

    Vertices are returned in the prototype-local frame, in millimeters, with
    REVERSED faces flipped so winding is consistent.
    """
    BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION, False, ANGULAR_DEFLECTION, True)

    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    vertex_offset = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        face_loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, face_loc)
        if tri is None:
            exp.Next()
            continue

        face_trsf = face_loc.Transformation()
        # Build a 4x4 to transform face-local nodes into prototype-local frame.
        face_m = np.eye(4)
        for r in range(3):
            for c in range(4):
                face_m[r, c] = face_trsf.Value(r + 1, c + 1)

        n_nodes = tri.NbNodes()
        verts = np.empty((n_nodes, 3), dtype=float)
        for i in range(1, n_nodes + 1):
            p = tri.Node(i)
            verts[i - 1] = (p.X(), p.Y(), p.Z())

        # Apply the face's location.
        verts_h = np.hstack([verts, np.ones((n_nodes, 1))])
        verts = (face_m @ verts_h.T).T[:, :3]

        reversed_face = face.Orientation() == TopAbs_REVERSED

        n_tri = tri.NbTriangles()
        tris = np.empty((n_tri, 3), dtype=np.int64)
        for i in range(1, n_tri + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            if reversed_face:
                n2, n3 = n3, n2
            tris[i - 1] = (n1 - 1 + vertex_offset,
                           n2 - 1 + vertex_offset,
                           n3 - 1 + vertex_offset)

        all_verts.append(verts)
        all_tris.append(tris)
        vertex_offset += n_nodes
        exp.Next()

    if not all_verts:
        return trimesh.Trimesh()

    V = np.vstack(all_verts)
    F = np.vstack(all_tris)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    # Drop duplicate verts and degenerate tris but keep face windings.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def main(step_path: str, out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        print("ReadFile failed", file=sys.stderr)
        return 1
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    if not reader.Transfer(doc):
        print("Transfer failed", file=sys.stderr)
        return 1
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    free = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free)
    if free.Length() != 1:
        print(f"Expected 1 root, got {free.Length()}", file=sys.stderr)
        return 1
    root = free.Value(1)
    print(f"Root: {label_name(root)!r}")

    comps = TDF_LabelSequence()
    shape_tool.GetComponents_s(root, comps, False)
    print(f"Components: {comps.Length()}")

    # Cache tessellated prototypes by their label entry string so shared
    # prototypes (e.g. the two identical 2040 extrusions) are meshed once.
    proto_cache: dict[str, trimesh.Trimesh] = {}
    records = []

    for j in range(1, comps.Length() + 1):
        comp = comps.Value(j)
        raw_name = label_name(comp)
        ref = TDF_Label()
        if not shape_tool.GetReferredShape_s(comp, ref):
            print(f"  [{j}] no referred shape, skipping", file=sys.stderr)
            continue
        proto_name = label_name(ref)
        loc = shape_tool.GetLocation_s(comp)
        placement = trsf_to_matrix_mm(loc)

        # Entry string uniquely identifies the prototype label.
        from OCP.TDF import TDF_Tool
        from OCP.TCollection import TCollection_AsciiString
        entry = TCollection_AsciiString()
        TDF_Tool.Entry_s(ref, entry)
        proto_key = entry.ToCString()

        if proto_key not in proto_cache:
            print(f"  meshing prototype {proto_name!r} (key={proto_key})")
            proto_shape = shape_tool.GetShape_s(ref)
            proto_cache[proto_key] = tessellate_prototype(proto_shape)

        mesh = proto_cache[proto_key].copy()
        if mesh.is_empty:
            print(f"  [{j}] empty mesh for proto {proto_name!r}", file=sys.stderr)
            continue

        stl_name = f"comp_{j:02d}.stl"
        mesh.export(out / stl_name)

        bbox = mesh.bounds  # (2,3) in mm
        records.append({
            "index": j,
            "stl": stl_name,
            "raw_name": raw_name,
            "proto_name": proto_name,
            "proto_key": proto_key,
            "transform_mm": placement.tolist(),
            "bbox_min_mm": bbox[0].tolist(),
            "bbox_max_mm": bbox[1].tolist(),
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
        })
        print(f"  [{j:2d}] -> {stl_name}  V={len(mesh.vertices):6d}  F={len(mesh.faces):6d}  "
              f"bbox_mm=({bbox[0][0]:+.1f},{bbox[0][1]:+.1f},{bbox[0][2]:+.1f}) "
              f"-> ({bbox[1][0]:+.1f},{bbox[1][1]:+.1f},{bbox[1][2]:+.1f})")

    sidecar = {
        "step_file": step_path,
        "units": "mm",
        "linear_deflection_mm": LINEAR_DEFLECTION,
        "angular_deflection_rad": ANGULAR_DEFLECTION,
        "components": records,
    }
    with open(out / "assembly.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(records)} STLs and assembly.json to {out}")
    return 0


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "raw/DDSM115 4WD-A.step"
    out = sys.argv[2] if len(sys.argv) > 2 else "meshes"
    sys.exit(main(step, out))
