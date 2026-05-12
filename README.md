# DDSM115 4WD-A MuJoCo model

A MuJoCo MJCF model of the Waveshare DDSM115 4WD-A robot, generated directly
from the manufacturer's open-source STEP assembly.

The pipeline downloads `DDSM115_4WD-A.zip` from Waveshare, parses the STEP
file with cadquery-ocp (OpenCascade bindings), tessellates each unique part
prototype into an STL, and emits an MJCF that uses one mesh per asset plus a
cylinder collider per wheel.

## Layout

```
raw/                          (gitignored) downloaded STEP archive + extract
meshes/                       (gitignored) comp_NN.stl + assembly.json
assets/                       18 ASCII-named STLs referenced by the MJCF
ddsm115_4wd.xml               the MJCF
inspect_step.py               dump assembly tree from the STEP
step_to_meshes.py             STEP -> per-component STL + assembly.json
build_model.py                meshes/ + assembly.json -> MJCF + assets/
test_model.py                 smoke-test the compiled model (forward + yaw)
Makefile                      `make` targets raw, meshes, model, test
```

## Quick start

```
pip install cadquery-ocp trimesh mujoco numpy
make test
```

`make all` (default `model`) runs the full pipeline; `make test` also loads
the MJCF in MuJoCo and exercises forward-drive and yaw episodes.

## STEP assembly

The 4WD-A STEP is a single-root assembly with 18 components in millimeters:

  * 4 DDSM115 wheel motors at chassis (±80, ±100, 0) mm
  * 4 CNC corner brackets (Chinese name `CNC链接件` with mirror variants)
  * 4 aluminum 2020 extrusions
  * 4 side supports (`侧支持`)
  * top plate (`盖板`) and bottom plate (`零部件25`)

`inspect_step.py` walks `STEPCAFControl_Reader` -> `GetFreeShapes` ->
`GetComponents` to print the tree.

`step_to_meshes.py` tessellates each unique prototype once with
`BRepMesh_IncrementalMesh(linear=0.3 mm, angular=0.5 rad)`, walks
`TopExp_Explorer(TopAbs_FACE)`, accumulates the triangulations into one
`trimesh.Trimesh` in the prototype-local frame, and reverses winding for
faces whose orientation is `TopAbs_REVERSED`. One STL is written per
component. `assembly.json` records each component's `raw_name`, `proto_name`,
4x4 placement matrix in mm, and local-frame bbox.

## MJCF conventions

  * `compiler angle=radian autolimits=true`, mesh `scale="0.001 0.001 0.001"`
    converts mm -> meters.
  * Chassis frame: **+Y is forward**, **+X is right**, **+Z is up**.
  * `base_link` is at world `(0, 0, wheel_radius + 0.001)` with a freejoint.
  * Non-wheel components are attached as fixed mesh geoms on `base_link` with
    `pos`/`quat` taken from each component's 4x4 placement.
  * Each wheel is a child body of `base_link` at the wheel's assembly
    pos/quat, with a hinge joint on `axis="0 1 0"` (wheel-local Y, which is
    the real axle), a visual mesh, and a thin cylinder collider rotated by
    `quat="0.7071068 0.7071068 0 0"` so its axis aligns with body-local Y.
    The cylinder, not the mesh, carries contacts.

### Default classes

| class            | type     | contype | density | friction          |
|------------------|----------|---------|---------|-------------------|
| `chassis`        | mesh     | 1       | 2000    | (default)         |
| `wheel_visual`   | mesh     | 0       | 600     | (default)         |
| `wheel_collision`| cylinder | 1       | 0       | 1.2 0.01 0.001    |

The DDSM115 hub motors are heavy (~0.4 kg each per datasheet) and the frame
is aluminum, so the densities are picked to land the total mass near the
real-robot 3.7 kg: the wheel mesh -- which already includes the motor
housing -- carries the wheel-assembly mass via `wheel_visual` density, and
the cylinder collider is mass-less (it exists only for stable contact).

### Drive convention

The two right wheels (X = +80) have their axle pointing along chassis -X,
while the two left wheels (X = -80) point along +X. To unify the sign
convention, the two left motors get `gear="-1"` so that `ctrl=+1` on all
four motors drives the robot forward (+Y). For a left turn:

```
ctrl[wheel_fr] = +1
ctrl[wheel_br] = +1
ctrl[wheel_fl] = -1
ctrl[wheel_bl] = -1
```

### Sensors

  * one `jointvel` per wheel hinge
  * `framepos` + `framequat` on `base_link`

## Validation

`test_model.py` checks:

  * `nq == 11` (free joint 7 + 4 wheel hinges) and `nu == 4`
  * wheel radius measured at the collider: ~0.0503 m
  * settle for 1 s under gravity -- base z stays near the wheel radius
  * `ctrl=[1,1,1,1]` -> base translates along +Y
  * right=+1, left=-1 -> positive yaw rate about +Z

With the chosen densities the compiled model lands at ~3.74 kg: chassis
~2.07 kg, each wheel assembly ~0.42 kg. The cylinder collider is mass-less;
each wheel body's mass and inertia come from the visual mesh, which is
asymmetric along the axle (the motor housing sits behind the wheel disc)
and therefore correctly off-center.

## Gotchas worth remembering

  * The four wheel local-Y axes do not all point the same direction in the
    chassis frame. `gear="-1"` on the two left motors is what unifies the
    sign convention.
  * Component names contain CJK characters; `build_model.py` slugs them to
    ASCII roles before any STL is renamed or referenced in the MJCF.
  * The chassis frame has Z = 0 at the wheel hub plane, so `base_link` world
    Z must be at least `wheel_radius` for the wheels to touch the floor.
    We use `wheel_radius + 0.001` for a hair of clearance.
  * `gp_Trsf` rows are 1-indexed: `trsf.Value(r+1, c+1)`.
