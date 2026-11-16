"""Unit tests for the parts that run without a GPU:  python -m unittest discover -s tests -v"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from evaluation.restoration_metrics import restoration_metrics
from evaluation.segmentation_metrics import binary_segmentation_metrics
from model.diffusion_model import canny_map, classical_inpaint, work_size
from model.inference import restore_mural
from model.postprocess import composite, feather, match_color, match_texture, postprocess, sharpen_edges, super_resolve
from tools.synthetic_damage import generate_synthetic_damage, validate_pair
from utils.mask_utils import apply_mask_addition, apply_mask_removal, mask_from_canvas_rgba, refine_predicted_mask


def toy_mural(seed=0, size=256):
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), np.uint8)
    for c in range(3):
        base[..., c] = np.linspace(70 + 20 * c, 200 - 10 * c, size)[None, :]
    base = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    base[60:120, 60:120] = (150, 60, 40)
    return Image.fromarray(base)


def blob_mask(size=256):
    m = np.zeros((size, size), np.uint8)
    m[100:140, 150:200] = 255
    m[30:36, 30:36] = 255
    return Image.fromarray(m)


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.img, self.mask = toy_mural(), blob_mask()
        self.gen = classical_inpaint(self.img, self.mask)

    def test_intact_pixels_bit_exact(self):
        out, info = postprocess(self.img, self.gen, self.mask, {"color": True, "texture": True, "sharpen": True})
        a, b = np.asarray(self.img).astype(int), np.asarray(out).astype(int)
        m = np.asarray(self.mask) > 127
        import cv2
        far = ~cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        self.assertEqual(np.abs(a - b)[far].max(), 0)
        self.assertIn("region_color_match", info["stages"])

    def test_feather_range_and_inside(self):
        m = np.asarray(self.mask) > 127
        f = feather(m, 6)
        self.assertTrue(((f >= 0) & (f <= 1)).all())
        self.assertTrue((f[m] == 1).all())

    def test_composite_keeps_outside(self):
        gen = Image.new("RGB", self.img.size, (0, 255, 0))
        out = np.asarray(composite(self.img, gen, self.mask, 4))
        self.assertTrue((out[0, 0] == np.asarray(self.img)[0, 0]).all())
        self.assertTrue((out[120, 170] == (0, 255, 0)).all())

    def test_color_match_moves_toward_ring(self):
        gen = Image.new("RGB", self.img.size, (255, 0, 255))      # blatantly wrong colour
        out = match_color(gen, self.img, self.mask, strength=1.0)
        before = np.abs(np.asarray(gen)[120, 170].astype(int) - np.asarray(self.img)[120, 170].astype(int)).sum()
        after = np.abs(np.asarray(out)[120, 170].astype(int) - np.asarray(self.img)[120, 170].astype(int)).sum()
        self.assertLess(after, before)

    def test_texture_and_sharpen_shapes(self):
        self.assertEqual(match_texture(self.gen, self.img, self.mask).size, self.img.size)
        self.assertEqual(sharpen_edges(self.gen, self.mask).size, self.img.size)

    def test_super_resolution_doubles(self):
        out, method = super_resolve(self.img, 2)
        self.assertEqual(out.size, (512, 512))
        self.assertTrue(method)


class PipelineTests(unittest.TestCase):
    def test_classical_pipeline_end_to_end(self):
        img, mask = toy_mural(), blob_mask()
        res = restore_mural(img, mask, config_dict={"restore_mode": "classical"})
        self.assertEqual(res.restored_image.size, img.size)
        self.assertIn("final", res.stages)
        self.assertEqual(res.mode, "classical")

    def test_post_disabled(self):
        img, mask = toy_mural(), blob_mask()
        res = restore_mural(img, mask, config_dict={"restore_mode": "classical", "post": False})
        self.assertEqual(res.metrics["post_stages"], ["composite"])

    def test_work_size_multiple_of_8(self):
        for size in [(512, 512), (1024, 768), (333, 777)]:
            w, h = work_size(size, "cpu")
            self.assertEqual((w % 8, h % 8), (0, 0))
            self.assertLessEqual(max(w, h), 512)


class MaskTests(unittest.TestCase):
    def test_refine_removes_specks_and_dilates(self):
        out = np.asarray(refine_predicted_mask(blob_mask(), close_px=0, min_area=100, dilate_px=3)) > 127
        self.assertFalse(out[30:36, 30:36].any())          # speck removed
        self.assertTrue(out[98, 150])                       # dilated beyond original edge

    def test_add_remove(self):
        base = blob_mask()
        layer = Image.new("L", base.size, 0)
        layer.paste(255, (0, 0, 20, 20))
        added = np.asarray(apply_mask_addition(base, layer)) > 127
        self.assertTrue(added[5, 5])
        removed = np.asarray(apply_mask_removal(base, layer)) > 127
        self.assertFalse(removed[5, 5])

    def test_canvas_rgba(self):
        rgba = np.zeros((64, 64, 4), np.uint8)
        rgba[10:20, 10:20] = (255, 0, 0, 255)
        m = np.asarray(mask_from_canvas_rgba(rgba, (128, 128)))
        self.assertTrue(m[30, 30] == 255 and m[100, 100] == 0)


class MetricTests(unittest.TestCase):
    def test_identical_images(self):
        img, mask = toy_mural(), blob_mask()
        r = restoration_metrics(img, img, mask, with_lpips=False)
        self.assertEqual(r["ssim"], 1.0)
        self.assertEqual(r["psnr_masked"], float("inf"))

    def test_segmentation_metrics(self):
        a = np.zeros((10, 10), bool); a[:5] = True
        b = np.zeros((10, 10), bool); b[:5, :5] = True
        r = binary_segmentation_metrics(b, a)
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertAlmostEqual(r["dice"], 2 / 3, places=5)


class MiscTests(unittest.TestCase):
    def test_synthetic_damage_valid(self):
        clean = np.asarray(toy_mural(size=384))
        for s in range(5):
            dam, mask = generate_synthetic_damage(clean, seed=s)
            validate_pair(clean, dam, mask)
            self.assertEqual(mask.shape, clean.shape[:2])

    def test_canny_map(self):
        self.assertEqual(canny_map(toy_mural()).mode, "L")

    def test_registry(self):
        self.assertEqual(list(config.SEGMENTATION_MODELS), ["attention_unet"])
        self.assertEqual(config.SEGMENTATION_MODELS["attention_unet"]["path"].name, "attention_unet.keras")

    def test_lora_key_prefix(self):
        import json, struct
        for p in (config.LORA_PATH, config.CANNY_LORA_PATH):
            if p.is_file():
                raw = open(p, "rb").read(); n = struct.unpack("<Q", raw[:8])[0]
                keys = [k for k in json.loads(raw[8:8 + n]) if k != "__metadata__"]
                self.assertTrue(all(k.startswith("unet.") for k in keys))
                self.assertEqual(len(keys), 256)


if __name__ == "__main__":
    unittest.main()
