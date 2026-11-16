"""Streamlit app - AI-based mural restoration (Attention U-Net -> damage-aware -> 3 diffusion stages -> post-processing).

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import os

# Kaggle / shared-GPU mode: keep TensorFlow (segmentation) on the CPU so PyTorch gets the whole GPU,
# and switch off peft's torchao hook (Kaggle's torchao version is too old). Harmless elsewhere.
if os.environ.get("MURAL_TF_CPU") == "1":
    try:
        import tensorflow as _tf
        _tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
try:
    import peft.tuners.lora.torchao as _pt
    _pt.is_torchao_available = lambda: False
except Exception:
    pass

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import config
from model.damage_detector import predict_damage_mask, threshold_mask
from model.diffusion_model import diffusion_available, resolve_device
from model.inference import restore_mural
from utils.mask_utils import (apply_mask_addition, apply_mask_removal, compute_damage_pixel_stats,
                              create_mask_overlay, refine_predicted_mask)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # editor is optional
    streamlit_image_coordinates = None

st.set_page_config(page_title="Mural Restoration AI", page_icon="🏛️", layout="wide")

P = config.HERITAGE_PALETTE
st.markdown(
    f"""<style>
    .stApp {{ background:{P['light_bg']}; }}
    h1,h2,h3 {{ color:{P['primary']}; }}
    .stButton>button[kind="primary"] {{ background:{P['primary']}; border:0; }}
    div[data-testid="stMetricValue"] {{ color:{P['primary']}; }}
    .badge {{ display:inline-block; padding:2px 10px; border-radius:12px; background:{P['secondary']}22;
              border:1px solid {P['secondary']}; font-size:0.8rem; margin-right:6px; }}
    </style>""",
    unsafe_allow_html=True,
)

DISPLAY_W = 560           # width of the click-editor canvas (px)


# ------------------------------------------------------------------------ helpers
def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sample_files():
    return sorted(p for p in config.SAMPLE_DIR.glob("*") if p.suffix.lower().lstrip(".") in config.SUPPORTED_FORMATS)


def mask_hash(mask: Image.Image) -> str:
    return hashlib.sha1(np.asarray(mask.convert("L")).tobytes()).hexdigest()[:10]


def cache_path(stem: str, mode: str) -> Path:
    return config.CACHE_DIR / f"{stem}__{mode}.png"


def reset_state(source_id: str):
    keep = {"source_id": source_id}
    st.session_state.clear()
    st.session_state.update(keep)


def save_run(image, result, mask_used, seg_stats, cfg):
    """Persist inputs/outputs/metadata under data/ so every demo run is reproducible."""
    run_id = f"mural_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    image.save(config.UPLOAD_DIR / f"{run_id}_original.png")
    mask_used.save(config.MASK_DIR / f"{run_id}_mask.png")
    result.restored_image.save(config.OUTPUT_DIR / f"{run_id}_restored.png")
    result.raw_restored_image.save(config.RAW_DIR / f"{run_id}_raw_restored.png")
    if result.canny_map is not None:
        result.canny_map.save(config.CANNY_DIR / f"{run_id}_canny.png")
    meta = {"run_id": run_id, "timestamp": datetime.now().isoformat(), "image_size": list(image.size),
            "segmentation": seg_stats, "settings": {k: v for k, v in cfg.items() if k != "post"},
            "post": cfg.get("post"), "metrics": result.metrics}
    (config.METADATA_DIR / f"{run_id}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return run_id


# ------------------------------------------------------------------ mask editor
def edit_mask_ui(image: Image.Image, base_mask: Image.Image) -> Image.Image:
    """Click-based human-in-the-loop editor: brush / rectangle / polygon, add or remove damage."""
    ss = st.session_state
    if streamlit_image_coordinates is None:
        st.warning("Install `streamlit-image-coordinates` to enable the mask editor.")
        return base_mask

    c1, c2, c3 = st.columns([1, 1, 1])
    action = c1.radio("Correction", ["Add damage", "Remove false detection"], key="edit_action")
    tool = c2.radio("Tool", ["Brush", "Rectangle (drag)", "Polygon (click points)"], key="edit_tool")
    brush = c3.slider("Brush radius (px)", 3, 60, 18, key="edit_brush")

    scale = image.width / DISPLAY_W
    disp_h = int(image.height / scale)
    edited = ss.get("edited_mask", base_mask).copy()
    poly = ss.setdefault("poly_pts", [])

    preview = create_mask_overlay(image, edited).resize((DISPLAY_W, disp_h))
    if poly:                                              # draw the polygon being built
        d = ImageDraw.Draw(preview)
        pts = [(x / scale, y / scale) for x, y in poly]
        d.line(pts + ([pts[0]] if len(pts) > 2 else []), fill=(255, 220, 0), width=2)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 220, 0))

    drag = tool.startswith("Rectangle")
    val = streamlit_image_coordinates(preview, width=DISPLAY_W, click_and_drag=drag, key=f"canvas_{ss.get('canvas_v', 0)}")

    layer = Image.new("L", image.size, 0)
    drawn = False
    stamp = (val or {}).get("unix_time")
    if val and stamp != ss.get("last_stamp"):
        ss["last_stamp"] = stamp
        d = ImageDraw.Draw(layer)
        if tool == "Brush" and "x" in val:
            x, y = val["x"] * scale, val["y"] * scale
            r = brush * scale / 1.0
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
            drawn = True
        elif drag and "x1" in val:
            x1, y1, x2, y2 = (val[k] * scale for k in ("x1", "y1", "x2", "y2"))
            if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
                d.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=255)
                drawn = True
        elif tool.startswith("Polygon") and "x" in val:
            poly.append((val["x"] * scale, val["y"] * scale))
            st.rerun()
    if drawn:
        edited = apply_mask_addition(edited, layer) if action == "Add damage" else apply_mask_removal(edited, layer)
        ss["edited_mask"] = edited
        ss["mask_edited"] = True
        st.rerun()

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Close polygon", disabled=len(poly) < 3):
        layer = Image.new("L", image.size, 0)
        ImageDraw.Draw(layer).polygon(poly, fill=255)
        ss["edited_mask"] = apply_mask_addition(edited, layer) if action == "Add damage" else apply_mask_removal(edited, layer)
        ss["mask_edited"] = True
        ss["poly_pts"] = []
        st.rerun()
    if b2.button("Clear polygon", disabled=not poly):
        ss["poly_pts"] = []
        st.rerun()
    if b3.button("Reset to automatic mask"):
        ss.pop("edited_mask", None)
        ss["mask_edited"] = False
        ss["poly_pts"] = []
        st.rerun()
    if b4.button("Clear whole mask"):
        ss["edited_mask"] = Image.new("L", image.size, 0)
        ss["mask_edited"] = True
        st.rerun()
    return ss.get("edited_mask", base_mask)


# ------------------------------------------------------------------------- sidebar
def sidebar():
    seg = config.SEGMENTATION_MODELS[config.DEFAULT_SEGMENTATION_MODEL]
    dev = resolve_device()
    can_diffuse = diffusion_available()
    with st.sidebar:
        st.header("1 · Segmentation")
        st.markdown(f"**Model:** {seg['label']}")
        if not Path(seg["path"]).is_file():
            st.error(f"Model file not found: models/{Path(seg['path']).name}")
            st.stop()
        thr = st.slider("Damage threshold", 0.05, 0.95, float(seg["threshold"]), 0.05,
                        help="Chosen on validation data (models/thresholds.json). Lower = more recall, more false positives.")
        refine = st.checkbox("Clean mask for diffusion (close, de-speckle, dilate)", True)
        dil = st.slider("Mask margin (px)", 0, 20, 6) if refine else 0

        st.header("2 · Damage-aware module")
        crack_backend = st.radio("Thin cracks", ["diffusion", "classical"], horizontal=True,
                                 format_func=lambda k: {"diffusion": "light diffusion", "classical": "classical only"}[k],
                                 help="Each region is classified as crack / flake / paint loss; its denoising strength "
                                      "comes from the damage type and the Attention U-Net severity.")

        st.header("3 · Diffusion stages")
        if can_diffuse:
            mode = "full"
            st.caption("Stage 1: SD 1.5 inpainting + mural LoRA  \nStage 2: Canny ControlNet + Canny LoRA  \n"
                       "Final diffusion: low-strength blending pass")
        else:
            mode = "classical"
            st.info("torch/diffusers not installed - classical preview only (use pre-computed results for diffusion).")
        st.caption(f"Device: **{dev.upper()}**" + ("  - expect several minutes per image on CPU." if dev == "cpu" and can_diffuse else ""))
        for name, path in (("Mural LoRA", config.LORA_PATH), ("Canny LoRA", config.CANNY_LORA_PATH)):
            st.caption(("✅ " if path.is_file() else "❌ missing: ") + f"{name} ({path.name})")
        steps = st.slider("Diffusion steps", 10, 75, 30 if dev != "cpu" else 20, 5)
        guidance = st.slider("Guidance scale", 1.0, 15.0, config.DEFAULT_GUIDANCE_SCALE, 0.5)
        seed = int(st.number_input("Seed", min_value=0, value=config.DEFAULT_SEED, step=1))
        stage2_strength = st.slider("Stage 2 strength", 0.3, 0.9, config.STAGE2_STRENGTH, 0.05)
        final_strength = st.slider("Final diffusion strength", 0.1, 0.6, config.FINAL_STRENGTH, 0.05)
        prompt = st.text_area("Prompt", config.DEFAULT_PROMPT)

        st.header("4 · Post-processing")
        col = st.checkbox("Region-wise colour matching", True)
        tex = st.checkbox("Texture matching", True)
        shp = st.checkbox("Edge-aware sharpening", True)
        sr = st.checkbox("Super-resolution x2", False)

        st.header("Demo")
        use_cache = st.checkbox("Use pre-computed result when available", True,
                                help="For CPU-only machines: tools/precompute_demo.py stores results for the sample images.")

    post = {"color": col, "texture": tex, "sharpen": shp, "super_res": sr}
    cfg = {"restore_mode": mode, "steps": steps, "guidance_scale": guidance, "seed": seed, "prompt": prompt,
           "use_lora": True, "stage2_strength": stage2_strength, "final_strength": final_strength,
           "refine_mask": refine, "dilate_px": dil, "post": post,
           "damage_aware": True, "crack_backend": crack_backend, "threshold": thr}
    return config.DEFAULT_SEGMENTATION_MODEL, thr, cfg, use_cache


# --------------------------------------------------------------------------- main
def main():
    st.title("🏛️ Mural Restoration AI")
    st.caption("Attention U-Net segmentation  →  damage-aware module  →  Stage 1 (SD + mural LoRA)  →  "
               "Stage 2 (Canny ControlNet + Canny LoRA)  →  final diffusion  →  post-processing")
    model_key, thr, cfg, use_cache = sidebar()
    ss = st.session_state

    src_col1, src_col2 = st.columns([2, 1])
    uploaded = src_col1.file_uploader("Upload a damaged mural", type=config.SUPPORTED_FORMATS)
    samples = sample_files()
    sample = src_col2.selectbox("…or pick a sample", ["-"] + [p.name for p in samples])
    if uploaded is None and sample == "-":
        st.info("Upload a mural image or choose a sample to begin.")
        return
    if uploaded is not None:
        data = uploaded.getvalue()
        image, stem = Image.open(io.BytesIO(data)).convert("RGB"), Path(uploaded.name).stem
        source_id = hashlib.sha1(data).hexdigest()
    else:
        path = config.SAMPLE_DIR / sample
        image, stem = Image.open(path).convert("RGB"), path.stem
        source_id = f"sample:{sample}"
    if ss.get("source_id") != source_id:
        reset_state(source_id)

    st.image(image, caption=f"Original ({image.width} × {image.height})", width=DISPLAY_W)

    # ---- 1. segmentation
    if st.button("1. Segment damage", type="primary", width="stretch"):
        with st.spinner(f"Running {config.SEGMENTATION_MODELS[model_key]['label']}…"):
            _, prob, _ = predict_damage_mask(image, threshold=thr, model_key=model_key)
        ss.update(prob=prob, seg_model=model_key)
        ss.pop("edited_mask", None)
        ss["mask_edited"] = False
        ss.pop("result", None)

    if ss.get("prob") is None:
        return

    # threshold is applied live to the stored probability map (no re-inference)
    auto_mask = threshold_mask(ss["prob"], thr)
    working = ss.get("edited_mask", auto_mask) if ss.get("mask_edited") else auto_mask
    stats = compute_damage_pixel_stats(working)
    seg_stats = {"model_key": ss["seg_model"], "threshold": thr, **stats, "mask_edited": bool(ss.get("mask_edited"))}

    c1, c2 = st.columns(2)
    c1.image(create_mask_overlay(image, working), caption="Detected / edited damage", width="stretch")
    c2.image(working, caption="Mask", width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Damage coverage", f"{stats['damage_percentage']:.2f}%")
    m2.metric("Damaged pixels", f"{stats['damage_pixels']:,}")
    m3.metric("Mask", "edited by conservator" if ss.get("mask_edited") else "automatic")

    with st.expander("✏️ Human-in-the-loop: review and correct the mask", expanded=False):
        if not ss.get("mask_edited"):
            ss["edited_mask"] = auto_mask
        working = edit_mask_ui(image, working if ss.get("mask_edited") else auto_mask)

    # ---- 2. restoration
    mode = cfg["restore_mode"]
    if st.button(f"2. Restore  ·  {config.RESTORE_MODES[mode]}", type="primary", width="stretch"):
        cached = cache_path(stem, "full")          # pre-computed full-pipeline result (GPU / Kaggle)
        if use_cache and cached.is_file() and not ss.get("mask_edited") and uploaded is None:
            final = Image.open(cached).convert("RGB")
            stages = {}
            for k in ("types", "prefill", "stage1", "stage2", "final_diffusion", "final"):
                f = config.CACHE_DIR / f"{stem}__full__{k}.png"
                if f.is_file():
                    stages[k] = Image.open(f).convert("RGB")
            meta_f = config.CACHE_DIR / f"{stem}__full__metrics.json"
            meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {}
            ss["result"] = {"final": final, "cached": True, "stages": stages,
                            "metrics": {"source": "pre-computed on GPU", "mode": "full", **meta}}
        else:
            with st.spinner("Restoring… (diffusion on CPU can take several minutes)"):
                t0 = time.time()
                # pixels the conservator added by hand count as fully severe damage
                prob_eff = np.where(np.asarray(working.convert("L")) > 127, np.maximum(ss["prob"], 
                                    np.where(np.asarray(auto_mask) > 127, 0.0, 1.0)), ss["prob"])
                res = restore_mural(image, working, config_dict=cfg, prob_map=prob_eff)
            run_id = save_run(image, res, res.refined_mask, seg_stats, cfg)
            ss["result"] = {"final": res.restored_image, "cached": False, "res": res, "run_id": run_id,
                            "metrics": res.metrics, "elapsed": time.time() - t0}

    out = ss.get("result")
    if out is None:
        return
    st.subheader("Result")
    if out["cached"]:
        st.markdown('<span class="badge">pre-computed result</span>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.image(image, caption="Before", width="stretch")
    b.image(out["final"], caption="Restored", width="stretch")
    st.download_button("Download restored mural", png_bytes(out["final"]), "restored_mural.png", "image/png", width="stretch")

    res = out.get("res")
    if res is not None:
        tabs = st.tabs(["Pipeline stages", "Damage analysis", "Run details"])
        with tabs[0]:
            names = [("prefill", "Classical pre-fill"), ("stage1", "Stage 1 · SD + mural LoRA"),
                     ("stage2", "Stage 2 · Canny ControlNet + Canny LoRA"), ("final_diffusion", "Final diffusion"),
                     ("final", "After post-processing")]
            shown = [(k, t) for k, t in names if k in res.stages]
            cols = st.columns(len(shown) + (1 if res.canny_map is not None else 0))
            for col, (k, t) in zip(cols, shown):
                col.image(res.stages[k], caption=t, width="stretch")
            if res.canny_map is not None:
                cols[-1].image(res.canny_map, caption="Canny edges (100/200)", width="stretch")
        with tabs[1]:
            if "types" in res.stages:
                c1, c2 = st.columns([3, 2])
                c1.image(res.stages["types"], caption="Damage types: red = crack, orange = flake, teal = paint loss",
                         width="stretch")
                c2.json(res.metrics.get("damage_summary", {}))
                c2.caption("Strength tiers run light → heavy; each diffusion call runs strength × steps.")
                st.dataframe([{k: r[k] for k in ("id", "type", "area_pct", "thickness_px", "elongation", "severity",
                                                   "strength", "tier")} for r in res.metrics.get("regions", [])],
                             width="stretch", hide_index=True)
            else:
                st.info("No damage analysis for this run.")
        with tabs[2]:
            st.json({k: v for k, v in {**out["metrics"], "saved_as": out.get("run_id")}.items() if k != "regions"})
    elif out.get("stages"):
        tabs = st.tabs(["Pipeline stages", "Damage analysis", "Run details"])
        with tabs[0]:
            names = [("prefill", "Classical pre-fill"), ("stage1", "Stage 1 · SD + mural LoRA"),
                     ("stage2", "Stage 2 · Canny ControlNet + Canny LoRA"), ("final_diffusion", "Final diffusion"),
                     ("final", "After post-processing")]
            shown = [(k, t) for k, t in names if k in out["stages"]]
            for col, (k, t) in zip(st.columns(len(shown)), shown):
                col.image(out["stages"][k], caption=t, width="stretch")
        with tabs[1]:
            if "types" in out["stages"]:
                c1, c2 = st.columns([3, 2])
                c1.image(out["stages"]["types"], caption="Damage types: red = crack, orange = flake, teal = paint loss",
                         width="stretch")
                c2.json(out["metrics"].get("damage_summary", {}))
                if out["metrics"].get("regions"):
                    st.dataframe([{k: r[k] for k in ("id", "type", "area_pct", "thickness_px", "elongation", "severity",
                                                       "strength", "tier")} for r in out["metrics"]["regions"]],
                                 width="stretch", hide_index=True)
        with tabs[2]:
            st.json({k: v for k, v in out["metrics"].items() if k != "regions"})
    else:
        st.json(out["metrics"])


if __name__ == "__main__":
    main()
