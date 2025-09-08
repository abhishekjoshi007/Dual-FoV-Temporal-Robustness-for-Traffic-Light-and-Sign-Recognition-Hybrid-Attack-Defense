#!/usr/bin/env python3
import os, json, math, random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

def ensure_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont
    except ImportError:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont

Image, ImageDraw, ImageFont = ensure_pillow()

# ---- config ----
DATASET_ROOT = "/Volumes/Untitled/Dataset"
OUT_DIR = "figs"
MAKE_PANEL = True
CELL_WIDTH = 600
ROW_LABEL_W = 180
COL_LABEL_H = 80
FONT_SCALE = 28

# 2D selection constraints
MIN_AREA_RATIO_TL = 0.0000   # set to 0.0 to avoid filtering out small boxes
MIN_AREA_RATIO_TS = 0.0000
CENTER_BIAS = 0.3            # 0..1
ALLOW_3D_FALLBACK = True     # if no 2D frames with TL+TS, pick by 3D presence

ODDS = ["highway", "night", "rainy", "urban"]
CAMERAS = {"F_MIDRANGECAM_C": "mid", "F_LONGRANGECAM_C": "long"}
ROW_LABELS = {"F_MIDRANGECAM_C": "Mid-range", "F_LONGRANGECAM_C": "Long-range"}
COL_LABELS = {"highway": "Highway", "night": "Night", "rainy": "Rainy", "urban": "Urban"}

# ---- io helpers ----
def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def _nonempty(p: Path) -> bool:
    try:
        d = load_json(p)
        return (isinstance(d, list) and len(d) > 0) or (isinstance(d, dict) and len(d) > 0)
    except: return False

def list_frame_ids(dir_path: Path) -> List[int]:
    return sorted([int(p.stem.split("_")[1]) for p in dir_path.glob("frame_*.json") if p.is_file()])

def img_path_for(seq: Path, camera: str, fid: int) -> Optional[Path]:
    cam_dir = seq/"sensor"/"camera"/camera
    for ext in (".jpg",".jpeg",".png"):
        q = cam_dir/f"{camera}_{fid:07d}{ext}"
        if q.exists(): return q
    m = list(cam_dir.glob(f"{camera}_{fid:07d}.*"))
    return m[0] if m else None

# ---- 2D-based selection ----
def frames_with_both_2d(seq: Path) -> List[int]:
    tl_dir = seq/"traffic_light"/"box"/"2d_body"
    ts_dir = seq/"traffic_sign"/"box"/"2d_body"
    if not tl_dir.exists() or not ts_dir.exists(): return []
    tl_ids = {int(p.stem.split("_")[1]) for p in tl_dir.glob("frame_*.json") if _nonempty(p)}
    ts_ids = {int(p.stem.split("_")[1]) for p in ts_dir.glob("frame_*.json") if _nonempty(p)}
    return sorted(tl_ids & ts_ids)

def load_2d_boxes(seq: Path, obj: str, fid: int) -> List[Dict]:
    p = seq/obj/"box"/"2d_body"/f"frame_{fid:07d}.json"
    if not p.exists(): return []
    try:
        d = load_json(p)
        return d if isinstance(d, list) else []
    except: return []

def clamp(x,a,b): return max(a, min(b, x))

def score_frame(img_w: int, img_h: int, tl_boxes: List[Dict], ts_boxes: List[Dict]) -> float:
    def box_score(b):
        x1,y1,x2,y2 = b
        w, h = max(0, x2-x1), max(0, y2-y1)
        area = w*h
        cx, cy = x1 + w/2.0, y1 + h/2.0
        nx, ny = (cx/img_w - 0.5), (cy/img_h - 0.5)
        center = 1.0 - clamp((nx*nx + ny*ny)**0.5 / 0.75, 0.0, 1.0)
        return (area/(img_w*img_h))*(1.0 - CENTER_BIAS) + center*CENTER_BIAS
    if not tl_boxes or not ts_boxes: return -1.0
    tl_best = max((box_score(r["bbox"]) for r in tl_boxes if "bbox" in r), default=-1.0)
    ts_best = max((box_score(r["bbox"]) for r in ts_boxes if "bbox" in r), default=-1.0)
    return (tl_best + ts_best) / 2.0

def passes_min_area(img_w:int,img_h:int, tl_boxes:List[Dict], ts_boxes:List[Dict]) -> bool:
    img_area = img_w*img_h
    def any_over(boxes, thr):
        if thr <= 0: return True
        for r in boxes:
            if "bbox" not in r: continue
            x1,y1,x2,y2 = r["bbox"]
            if (max(0,x2-x1)*max(0,y2-y1))/img_area >= thr: return True
        return False
    return any_over(tl_boxes, MIN_AREA_RATIO_TL) and any_over(ts_boxes, MIN_AREA_RATIO_TS)

def best_2d_example_for_seq(seq: Path, camera: str) -> Optional[Dict]:
    fids = frames_with_both_2d(seq)
    best = None; best_score = -1.0
    for fid in fids:
        img_p = img_path_for(seq, camera, fid)
        if not img_p: continue
        img = Image.open(img_p).convert("RGB")
        w,h = img.width, img.height
        tl = load_2d_boxes(seq, "traffic_light", fid)
        ts = load_2d_boxes(seq, "traffic_sign",  fid)
        if not passes_min_area(w,h,tl,ts): continue
        sc = score_frame(w,h,tl,ts)
        if sc > best_score:
            best_score = sc
            best = {"img": img_p, "fid": fid, "tl": tl, "ts": ts, "seq": seq, "box_type": "2d_body"}
    return best

# ---- 3D fallback (presence only) ----
def frames_with_both_3d(seq: Path) -> List[int]:
    tl_dir = seq/"traffic_light"/"box"/"3d_body"
    ts_dir = seq/"traffic_sign"/"box"/"3d_body"
    if not tl_dir.exists() or not ts_dir.exists(): return []
    tl_ids = {int(p.stem.split("_")[1]) for p in tl_dir.glob("frame_*.json") if _nonempty(p)}
    ts_ids = {int(p.stem.split("_")[1]) for p in ts_dir.glob("frame_*.json") if _nonempty(p)}
    return sorted(tl_ids & ts_ids)

def fallback_3d_example_for_seq(seq: Path, camera: str) -> Optional[Dict]:
    fids = frames_with_both_3d(seq)
    if not fids: return None
    # simple choice: middle frame
    fid = fids[len(fids)//2]
    img_p = img_path_for(seq, camera, fid)
    if not img_p: return None
    return {"img": img_p, "fid": fid, "tl": [], "ts": [], "seq": seq, "box_type": "3d_body"}

# ---- drawing/panel ----
def draw_boxes(img_path: Path, tl: List[Dict], ts: List[Dict]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("Arial.ttf", 18)
    except: font = ImageFont.load_default()
    for r in tl:
        if "bbox" not in r: continue
        x1,y1,x2,y2 = [int(v) for v in r["bbox"]]
        d.rectangle([x1,y1,x2,y2], outline=(0,200,0), width=3)
        d.text((x1+2, max(0,y1-18)), "TL", fill=(0,120,0), font=font)
    for r in ts:
        if "bbox" not in r: continue
        x1,y1,x2,y2 = [int(v) for v in r["bbox"]]
        d.rectangle([x1,y1,x2,y2], outline=(220,0,0), width=3)
        d.text((x1+2, max(0,y1-18)), "TS", fill=(160,0,0), font=font)
    return img

def save_img(img: Image.Image, dst: Path, max_w: int):
    if max_w and img.width > max_w:
        s = max_w/float(img.width)
        img = img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=95)

def text_size(draw, txt, font) -> Tuple[int,int]:
    if hasattr(draw,"textbbox"):
        l,t,r,b = draw.textbbox((0,0), txt, font=font)
        return r-l, b-t
    return font.getsize(txt)

def build_panel(paths: Dict[Tuple[int,int], Path], out_path: Path):
    try: font = ImageFont.truetype("Arial.ttf", FONT_SCALE)
    except: font = ImageFont.load_default()
    cells, max_h = {}, 0
    for (r,c), p in paths.items():
        if not (p and p.exists()): cells[(r,c)] = None; continue
        im = Image.open(p).convert("RGB")
        s = CELL_WIDTH/float(im.width)
        im = im.resize((CELL_WIDTH, int(im.height*s)), Image.LANCZOS)
        cells[(r,c)] = im
        max_h = max(max_h, im.height)
    cols, rows = len(ODDS), 2
    W = ROW_LABEL_W + cols*CELL_WIDTH
    H = COL_LABEL_H + rows*max_h
    panel = Image.new("RGB", (W,H), (255,255,255))
    dr = ImageDraw.Draw(panel)
    for ci, odd in enumerate(ODDS):
        lab = COL_LABELS[odd]; tw,th = text_size(dr, lab, font)
        dr.text((ROW_LABEL_W + ci*CELL_WIDTH + CELL_WIDTH//2 - tw//2, COL_LABEL_H//2 - th//2), lab, fill=(0,0,0), font=font)
    for ri, cam in enumerate(["F_MIDRANGECAM_C","F_LONGRANGECAM_C"]):
        lab = ROW_LABELS[cam]; tw,th = text_size(dr, lab, font)
        dr.text((ROW_LABEL_W//2 - tw//2, COL_LABEL_H + ri*max_h + max_h//2 - th//2), lab, fill=(0,0,0), font=font)
    for ri in range(2):
        for ci in range(cols):
            im = cells.get((ri,ci))
            x0 = ROW_LABEL_W + ci*CELL_WIDTH
            y0 = COL_LABEL_H + ri*max_h
            if im is None:
                ph = Image.new("RGB", (CELL_WIDTH, max_h), (230,230,230))
                dp = ImageDraw.Draw(ph); tw,th = text_size(dp,"MISSING",font)
                dp.text(((CELL_WIDTH-tw)//2,(max_h-th)//2),"MISSING",fill=(120,120,120),font=font)
                panel.paste(ph,(x0,y0))
            else:
                panel.paste(im,(x0,y0 + (max_h-im.height)//2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=95)

# ---- selection across dataset ----
def pick_example(root: Path, odd: str, camera: str) -> Optional[Dict]:
    odd_dir = root/odd
    if not odd_dir.exists(): return None
    # try 2D-first across sequences
    best = None; best_score = -1.0
    for seq in sorted([p for p in odd_dir.iterdir() if p.is_dir()]):
        ex = best_2d_example_for_seq(seq, camera)
        if not ex: continue
        img = Image.open(ex["img"]).convert("RGB")
        w,h = img.width, img.height
        sc = score_frame(w,h, ex["tl"], ex["ts"])
        if sc > best_score:
            best_score = sc; best = ex
    if best: return best
    if not ALLOW_3D_FALLBACK: return None
    # fallback to 3D presence if no 2D found anywhere
    for seq in sorted([p for p in odd_dir.iterdir() if p.is_dir()]):
        ex3d = fallback_3d_example_for_seq(seq, camera)
        if ex3d: return ex3d
    return None

def main():
    root = Path(DATASET_ROOT)
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

    selections: Dict[str, Dict] = {}
    for odd in ODDS:
        for cam, tag in CAMERAS.items():
            ex = pick_example(root, odd, cam)
            selections[f"{odd}_{tag}"] = ex
            print(f"[pick] {odd}/{cam} -> {(ex['img'], ex['box_type']) if ex else None}")

    name_map = {
        "highway_mid":"highway_mid.jpg","highway_long":"highway_long.jpg",
        "night_mid":"night_mid.jpg","night_long":"night_long.jpg",
        "rainy_mid":"rainy_mid.jpg","rainy_long":"rainy_long.jpg",
        "urban_mid":"urban_mid.jpg","urban_long":"urban_long.jpg",
    }

    for key, fn in name_map.items():
        ex = selections.get(key); dst = out/fn
        if ex is None:
            Image.new("RGB",(CELL_WIDTH,int(CELL_WIDTH*9/16)),(230,230,230)).save(dst,quality=95)
            continue
        if ex["box_type"] == "2d_body":
            img = draw_boxes(ex["img"], ex["tl"], ex["ts"])
        else:
            img = Image.open(ex["img"]).convert("RGB")
        if img.width > CELL_WIDTH:
            s = CELL_WIDTH/float(img.width)
            img = img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=95)

    if MAKE_PANEL:
        grid = {}
        for ci, odd in enumerate(ODDS):
            grid[(0,ci)] = out/name_map[f"{odd}_mid"]
            grid[(1,ci)] = out/name_map[f"{odd}_long"]
        build_panel(grid, out/"fig2_panel.png")
        print(f"[done] panel -> {out/'fig2_panel.png'}")

if __name__ == "__main__":
    main()
