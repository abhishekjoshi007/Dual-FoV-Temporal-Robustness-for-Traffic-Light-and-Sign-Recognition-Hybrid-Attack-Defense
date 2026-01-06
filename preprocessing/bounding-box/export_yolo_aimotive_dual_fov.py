#!/usr/bin/env python3
"""
Export 2D image bounding boxes by projecting aiMotive 3D cuboids into pixels.
- Supports F_MIDRANGECAM_C and F_LONGRANGECAM_C
- Uses calibration.json (intrinsics) and extrinsic_matrices.json (vehicle<->camera)
- Optionally uses egomotion2.json if object coords are in 'world' frame

Outputs:
- YOLO label files per image
- Visualization images with 2D rectangles (and optional 3D wireframe)

Defaults at the top can be edited to your local paths; CLI args override them.
"""

import json, math, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# =========================
# DEFAULTS (edit these)
# =========================
DEFAULTS = {
    "data_root": "/Users/abhishekjoshi/Documents/GitHub/Black-Box-Attack-Autonomus-Vechiles/Sample Dataset",
    "out_yolo":  "/Users/abhishekjoshi/Documents/GitHub/Black-Box-Attack-Autonomus-Vechiles/output/labels",
    "out_viz":   "/Users/abhishekjoshi/Documents/GitHub/Black-Box-Attack-Autonomus-Vechiles/output/viz",
    "odds":      ["urban","highway","night","rainy"],
    "draw_3d":   True,   # also draw 3D wireframe projection for QC
    "verbose":   True
}

# =========================
# Class mapping (edit for your classes)
# =========================
CLS = {
    "traffic_light": 0,
    "traffic_sign":  1
}

# ---------- Math helpers ----------
def rotx(a):  # roll
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float64)

def roty(a):  # pitch
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float64)

def rotz(a):  # yaw
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float64)

def euler_ypr(yaw, pitch, roll):
    # Yaw (Z) -> Pitch (Y) -> Roll (X); adjust if your dataset uses a different convention
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def make_box_corners(L, W, H):
    # centered at origin in object frame; X forward, Y left, Z up (adapt if needed)
    # corners in (X,Y,Z)
    x = L/2; y = W/2; z = H/2
    corners = np.array([
        [ x,  y,  z],
        [ x, -y,  z],
        [-x, -y,  z],
        [-x,  y,  z],
        [ x,  y, -z],
        [ x, -y, -z],
        [-x, -y, -z],
        [-x,  y, -z],
    ], dtype=np.float64)
    return corners

def load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

# ---------- Calibration loaders (robust to key variants) ----------
def load_intrinsics(calib_json: dict, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return K (3x3) and dist (k1,k2,p1,p2,k3) for cam_name.
    Tries to be robust to different JSON shapes.
    """
    K = None; dist = np.zeros((5,1), dtype=np.float64)
    # Try common patterns
    cams = calib_json.get("cameras") or calib_json.get("Cameras") or []
    # sometimes stored as dict keyed by name
    if isinstance(cams, dict):
        cams = [{**v, "name": k} for k,v in cams.items()]
    for c in cams:
        name = (c.get("name") or c.get("Name") or "").strip()
        if name != cam_name: 
            continue
        intr = c.get("intrinsic") or c.get("Intrinsics") or c.get("K") or {}
        # possibility: a flat list or fx/fy/cx/cy
        if isinstance(intr, dict):
            fx = intr.get("fx") or intr.get("Fx") or intr.get("f_x")
            fy = intr.get("fy") or intr.get("Fy") or intr.get("f_y")
            cx = intr.get("cx") or intr.get("Cx") or intr.get("c_x")
            cy = intr.get("cy") or intr.get("Cy") or intr.get("c_y")
            if all(v is not None for v in (fx,fy,cx,cy)):
                K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        elif isinstance(intr, list) and len(intr) in (4,9):
            if len(intr)==4:
                fx,fy,cx,cy = intr
                K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
            else:
                K = np.array(intr, dtype=np.float64).reshape(3,3)

        # distortion
        dct = c.get("distortion") or c.get("Distortion") or c.get("D") or {}
        if isinstance(dct, dict):
            k1 = dct.get("k1",0); k2=dct.get("k2",0); p1=dct.get("p1",0); p2=dct.get("p2",0); k3=dct.get("k3",0)
            dist = np.array([[k1],[k2],[p1],[p2],[k3]], dtype=np.float64)
        elif isinstance(dct, list) and len(dct)>=5:
            dist = np.array(dct[:5], dtype=np.float64).reshape(-1,1)

    if K is None:
        raise RuntimeError(f"Intrinsics for camera '{cam_name}' not found in calibration.json")
    return K, dist

def load_extrinsic(extr_json: dict, cam_name: str) -> np.ndarray:
    """
    Return 4x4 transform from VEHICLE -> CAMERA (T_cam_vehicle) if possible.
    Tries common key shapes.
    """
    T = None
    mats = extr_json.get("extrinsics") or extr_json.get("Extrinsics") or extr_json
    # could be {cam_name: {matrix: [...]}}
    if isinstance(mats, dict) and cam_name in mats:
        node = mats[cam_name]
        arr = node.get("matrix") or node.get("Matrix") or node
        arr = np.array(arr, dtype=np.float64).reshape(4,4)
        T = arr
    elif isinstance(mats, list):
        for node in mats:
            name = (node.get("name") or node.get("sensor") or "").strip()
            if name == cam_name:
                arr = node.get("matrix") or node.get("Matrix") or node.get("T") or node.get("value")
                if arr is not None:
                    T = np.array(arr, dtype=np.float64).reshape(4,4)
                    break
    if T is None:
        raise RuntimeError(f"Extrinsic for camera '{cam_name}' not found in extrinsic_matrices.json")
    return T

# ---------- Ego pose per frame (optional) ----------
def load_egomotion(ego_path: Path) -> Dict[str, np.ndarray]:
    """
    Parse egomotion2.json into a dict frame_idx -> world_T_vehicle (4x4).
    If not available, return empty dict; we’ll assume objects already in vehicle frame.
    """
    if not ego_path.exists():
        return {}
    data = load_json(ego_path)
    if data is None:
        return {}
    by_frame = {}
    # Try common structures:
    # [{"frame":"0000123","T_world_vehicle":[...]}]  or similar
    items = data.get("poses") or data.get("trajectory") or data
    if isinstance(items, list):
        for it in items:
            frame = None
            for k in ("frame","Frame","index","idx"):
                if k in it:
                    frame = f"{int(it[k]):07d}"
                    break
            if frame is None:
                # fallback: infer from timestamp field name like "frame_0001234"
                for k,v in it.items():
                    m = re.search(r"frame[_\-]?0*([0-9]+)", str(v))
                    if m:
                        frame = f"{int(m.group(1)):07d}"
                        break
            T = None
            for key in ("T_world_vehicle","world_T_vehicle","T","matrix"):
                if key in it:
                    arr = np.array(it[key], dtype=np.float64)
                    T = arr.reshape(4,4) if arr.size==16 else None
                    break
            if frame and T is not None:
                by_frame[frame] = T
    return by_frame

# ---------- Projection ----------
def project_points(K, dist, pts_cam):
    """
    pts_cam: Nx3 in camera frame. Returns Nx2 pixel coordinates (float).
    """
    rvec = np.zeros((3,1), dtype=np.float64)
    tvec = np.zeros((3,1), dtype=np.float64)
    pts = pts_cam.reshape(-1,1,3).astype(np.float64)
    img_pts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return img_pts.reshape(-1,2)

def transform(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """ Apply 4x4 to Nx3 points. """
    N = xyz.shape[0]
    homo = np.hstack([xyz, np.ones((N,1), dtype=np.float64)])
    out = (T @ homo.T).T
    return out[:, :3]

def inv(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]; t = T[:3,3:4]
    Rinv = R.T
    tinv = -Rinv @ t
    Tout = np.eye(4, dtype=np.float64)
    Tout[:3,:3] = Rinv; Tout[:3,3:4] = tinv
    return Tout

# ---------- Main per-frame processing ----------
def process_frame(json_path: Path,
                  seq_root: Path,
                  cam_name: str,
                  K: np.ndarray,
                  dist: np.ndarray,
                  T_cam_vehicle: np.ndarray,
                  world_T_vehicle_by_frame: Dict[str,np.ndarray],
                  out_yolo_root: Path,
                  out_viz_root: Path,
                  verbose: bool,
                  draw_3d: bool) -> int:

    # frame_0001234.json → 0001234
    m = re.search(r"frame[_\-]?0*([0-9]+)", json_path.stem)
    frame_idx = f"{int(m.group(1)):07d}" if m else json_path.stem

    # image path
    img_dir = seq_root / "sensor" / "camera" / cam_name
    img = None; img_path = None
    for ext in (".jpg",".jpeg",".png"):
        cand = img_dir / f"{cam_name}_{frame_idx}{ext}"
        if cand.exists():
            img_path = cand
            img = cv2.imread(str(cand))
            break
    if img is None:
        if verbose: print(f"[WARN] Missing image for {json_path}")
        return 0
    H, W = img.shape[:2]

    data = load_json(json_path)
    if data is None:
        if verbose: print(f"[WARN] Bad JSON: {json_path}")
        return 0

    objs = data.get("CapturedObjects") or data.get("objects") or data.get("annotations") or []
    if not isinstance(objs, list): 
        objs = []

    # Prepare transforms
    # If object is in 'vehicle' frame: cam_T_vehicle transforms points to camera
    cam_T_vehicle = T_cam_vehicle.copy()
    vehicle_T_cam = inv(cam_T_vehicle)

    # If object is in world frame: need world_T_vehicle(frame) too
    world_T_vehicle = world_T_vehicle_by_frame.get(frame_idx, None)
    if world_T_vehicle is None and verbose:
        print(f"[INFO] No world_T_vehicle for frame {frame_idx} (assuming objects in vehicle frame)")

    yolo_lines = []
    overlay = img.copy()
    num_boxes = 0

    for obj in objs:
        fam_guess = "traffic_sign" if "sign" in str(obj.get("ObjectType","")).lower() else "traffic_light"
        cid = CLS.get(fam_guess, 1)

        # Pull 3D box params
        def g(k): return obj.get(k) or obj.get(k.replace(" ", "")) or obj.get(k.lower().replace(" ","_"))
        L = float(g("BoundingBox3D Length") or 0)
        Wd= float(g("BoundingBox3D Width") or 0)
        Ht= float(g("BoundingBox3D Height") or 0)
        ox = float(g("BoundingBox3D Origin X") or 0)
        oy = float(g("BoundingBox3D Origin Y") or 0)
        oz = float(g("BoundingBox3D Origin Z") or 0)
        yaw   = math.radians(float(g("RotationYaw")   or 0))
        pitch = math.radians(float(g("RotationPitch") or 0))
        roll  = math.radians(float(g("RotationRoll")  or 0))

        if L<=0 or Wd<=0 or Ht<=0:
            continue

        # Build corners in object local frame, then rotate+translate to object frame
        corners = make_box_corners(L, Wd, Ht)           # Nx3
        R_obj   = euler_ypr(yaw, pitch, roll)           # 3x3
        corners_obj = (R_obj @ corners.T).T + np.array([[ox,oy,oz]], dtype=np.float64)

        # Decide which frame the origin is in
        # Heuristic: if we have world_T_vehicle, assume obj is in WORLD; else VEHICLE.
        # If your dataset encodes a flag, check it here instead.
        if world_T_vehicle is not None:
            world_T_obj = np.eye(4, dtype=np.float64)
            world_T_obj[:3,:3] = np.eye(3)  # already applied rotation above
            world_T_obj[:3, 3] = np.array([0,0,0], dtype=np.float64)  # corners_obj already in world coords if ox,oy,oz were world; adjust if needed

            # If ox,oy,oz are actually in VEHICLE coords, convert to WORLD:
            # corners_world = transform(world_T_vehicle, corners_vehicle)
            # If they are already WORLD, just pass them through.
            # We try both; prefer vehicle->world path (more common).
            corners_world = transform(world_T_vehicle, corners_obj)

            # CAMERA <- VEHICLE <- WORLD
            vehicle_T_world = inv(world_T_vehicle)
            cam_T_world = cam_T_vehicle @ vehicle_T_world
            corners_cam = transform(cam_T_world, corners_world)
        else:
            # assume corners_obj are in VEHICLE frame already
            corners_cam = transform(cam_T_vehicle, corners_obj)

        # Keep only points in front of camera
        in_front = corners_cam[:,2] > 0.1
        if in_front.sum() == 0:
            continue

        # Project to pixels
        img_pts = project_points(K, dist, corners_cam)
        xs, ys = img_pts[:,0], img_pts[:,1]
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)

        # Clamp to image
        xmin = max(0, min(W-1, xmin))
        xmax = max(0, min(W-1, xmax))
        ymin = max(0, min(H-1, ymin))
        ymax = max(0, min(H-1, ymax))
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue

        # Draw 2D rectangle
        p1 = (int(round(xmin)), int(round(ymin)))
        p2 = (int(round(xmax)), int(round(ymax)))
        cv2.rectangle(overlay, p1, p2, (0,255,0), 2)
        cv2.putText(overlay, f"{fam_guess}", (p1[0], max(0,p1[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # (Optional) draw 3D wireframe
        if draw_3d:
            # edges of a box
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a,b in edges:
                pa = (int(round(img_pts[a,0])), int(round(img_pts[a,1])))
                pb = (int(round(img_pts[b,0])), int(round(img_pts[b,1])))
                cv2.line(overlay, pa, pb, (255,0,0), 1, cv2.LINE_AA)

        # YOLO line
        cx = (xmin + xmax) / 2.0 / W
        cy = (ymin + ymax) / 2.0 / H
        ww = (xmax - xmin) / W
        hh = (ymax - ymin) / H
        yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
        num_boxes += 1

    # Write outputs
    if num_boxes > 0:
        rel = img_path.relative_to(seq_root / "sensor" / "camera" / cam_name)
        out_yolo = out_yolo_root / cam_name / rel.with_suffix(".txt")
        out_yolo.parent.mkdir(parents=True, exist_ok=True)
        (out_yolo).write_text("\n".join(yolo_lines))

        out_img = out_viz_root / cam_name / rel
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), overlay)

    return num_boxes

# ---------- Runner ----------
def run(data_root: Path, out_yolo: Path, out_viz: Path, odds: List[str], verbose: bool, draw_3d: bool):
    total_imgs = 0
    total_boxes = 0

    for odd in odds:
        odd_dir = data_root / odd
        if not odd_dir.exists():
            if verbose: print(f"[WARN] Missing ODD folder: {odd_dir}")
            continue

        # every sequence folder has: sensor/{calibration,camera,gnssins}, traffic_light/box/3d_body, traffic_sign/box/3d_body
        for seq_dir in sorted(odd_dir.iterdir()):
            if not seq_dir.is_dir(): continue

            cal_path = seq_dir / "sensor" / "calibration" / "calibration.json"
            ext_path = seq_dir / "sensor" / "calibration" / "extrinsic_matrices.json"
            ego_path = seq_dir / "sensor" / "gnssins" / "egomotion2.json"

            if not cal_path.exists() or not ext_path.exists():
                if verbose: print(f"[WARN] Missing calibration in {seq_dir} → skipping (need calibration/extrinsics for projection)")
                continue

            calib = load_json(cal_path)
            extr  = load_json(ext_path)
            if calib is None or extr is None:
                if verbose: print(f"[WARN] Bad calibration JSON under {seq_dir}")
                continue

            world_T_vehicle_by_frame = load_egomotion(ego_path)

            for cam_name in ("F_MIDRANGECAM_C", "F_LONGRANGECAM_C"):
                try:
                    K, dist = load_intrinsics(calib, cam_name)
                    T_cam_vehicle = load_extrinsic(extr, cam_name)
                except Exception as e:
                    if verbose: print(f"[WARN] {e}")
                    continue

                for fam in ("traffic_light","traffic_sign"):
                    box_dir = seq_dir / fam / "box" / "3d_body"
                    if not box_dir.exists(): 
                        continue
                    for jpath in sorted(box_dir.glob("frame_*.json")):
                        n = process_frame(jpath, seq_dir, cam_name, K, dist, T_cam_vehicle,
                                          world_T_vehicle_by_frame, Path(out_yolo), Path(out_viz),
                                          verbose, draw_3d)
                        if n>0: total_imgs += 1; total_boxes += n

    print("\n=== DONE ===")
    print(f"Images with labels: {total_imgs}")
    print(f"Total boxes      : {total_boxes}")
    print(f"YOLO labels to   : {out_yolo}")
    print(f"Visualizations   : {out_viz}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Project aiMotive 3D boxes to 2D image bboxes (dual-FoV)")
    ap.add_argument("--data_root", default=DEFAULTS["data_root"])
    ap.add_argument("--out_yolo",  default=DEFAULTS["out_yolo"])
    ap.add_argument("--out_viz",   default=DEFAULTS["out_viz"])
    ap.add_argument("--odds", nargs="+", default=DEFAULTS["odds"])
    ap.add_argument("--no_3d", action="store_true", help="Disable drawing 3D wireframe")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(Path(args.data_root),
        Path(args.out_yolo),
        Path(args.out_viz),
        args.odds,
        verbose=not args.quiet,
        draw_3d=(not args.no_3d))
