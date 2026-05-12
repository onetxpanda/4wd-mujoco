#!/usr/bin/env python3
"""Inspect the DDSM115 4WD-A STEP file: dump the top-level assembly tree.

Reports each free shape, its components (referred prototype + location), and
the prototype name/raw label. Used to confirm the 4WD-A structure (chassis +
4 wheel motors + 4 brackets + 4 extrusions + 4 side supports + 2 plates).
"""

from __future__ import annotations

import sys
from pathlib import Path

from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.IFSelect import IFSelect_RetDone


def label_name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return "<unnamed>"


def trsf_translation_mm(loc) -> tuple[float, float, float]:
    t = loc.Transformation()
    return (t.Value(1, 4), t.Value(2, 4), t.Value(3, 4))


def main(step_path: str) -> int:
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)

    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        print(f"ReadFile failed: {status}", file=sys.stderr)
        return 1

    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    if not reader.Transfer(doc):
        print("Transfer failed", file=sys.stderr)
        return 1

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    free = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free)
    print(f"Free shapes: {free.Length()}")

    for i in range(1, free.Length() + 1):
        root = free.Value(i)
        rname = label_name(root)
        is_assembly = shape_tool.IsAssembly_s(root)
        print(f"\nRoot [{i}] {rname!r}  is_assembly={is_assembly}")

        comps = TDF_LabelSequence()
        shape_tool.GetComponents_s(root, comps, False)
        print(f"  Components: {comps.Length()}")

        for j in range(1, comps.Length() + 1):
            comp = comps.Value(j)
            cname = label_name(comp)
            ref = TDF_Label()
            ok = shape_tool.GetReferredShape_s(comp, ref)
            proto_name = label_name(ref) if ok else "<no-ref>"
            loc = shape_tool.GetLocation_s(comp)
            tx, ty, tz = trsf_translation_mm(loc)
            print(
                f"   [{j:2d}] raw={cname!r:35s} proto={proto_name!r:25s} "
                f"pos_mm=({tx:+8.2f},{ty:+8.2f},{tz:+8.2f})"
            )


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "raw/DDSM115 4WD-A.step"
    if not Path(p).exists():
        print(f"Missing STEP file: {p}", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(p))
