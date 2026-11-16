import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import model.damage_aware as da
from model.inference import restore_mural


def scene():
    """512x512: one thin crack, one small flake, one large paint loss."""
    m = np.zeros((512, 512), np.uint8)
    cv2.line(m, (20, 40), (300, 90), 255, 4)                 # crack: 4 px wide, long
    cv2.circle(m, (420, 120), 14, 255, -1)                   # flake: ~0.24 % area, 28 px thick
    cv2.rectangle(m, (60, 250), (230, 450), 255, -1)         # loss: ~13 % area
    img = np.full((512, 512, 3), (170, 130, 90), np.uint8)
    cv2.circle(img, (256, 256), 180, (90, 40, 30), 6)
    return Image.fromarray(img), Image.fromarray(m)


class DamageAwareTests(unittest.TestCase):
    def test_classification(self):
        _, m = scene()
        regs = sorted(da.analyze_regions(m), key=lambda r: r.thickness_px)
        self.assertEqual([r.type for r in regs], ["crack", "flake", "loss"])

    def test_strength_order_and_tiers(self):
        _, m = scene()
        regs = {r.type: r for r in da.analyze_regions(m)}
        self.assertLess(regs["crack"].strength, regs["flake"].strength)
        self.assertLess(regs["flake"].strength, regs["loss"].strength)
        self.assertEqual({r.tier for r in regs.values()}, set(da.TIERS))

    def test_severity_from_probability(self):
        _, m = scene()
        mb = np.asarray(m) > 127
        low = np.where(mb, 0.55, 0.0); high = np.where(mb, 0.98, 0.0)
        s_low = {r.type: r.strength for r in da.analyze_regions(m, low)}
        s_high = {r.type: r.strength for r in da.analyze_regions(m, high)}
        for t in s_low:
            self.assertLessEqual(s_low[t], s_high[t])
        self.assertLess(s_low["flake"], s_high["flake"])

    def test_scale_invariance(self):
        img, m = scene()
        big = m.resize((1024, 1024), Image.Resampling.NEAREST)
        self.assertEqual(sorted(r.type for r in da.analyze_regions(m)), sorted(r.type for r in da.analyze_regions(big)))

    def test_tier_calls_and_routing(self):
        img, m = scene()
        calls = []

        def fake(image, mask, mode, cfg):
            calls.append((mode, cfg["strength"], int((np.asarray(mask) > 127).sum())))
            return {"stage1": Image.new("RGB", image.size, (10, 200, 10))}, {"backend": "fake", "execution_time_seconds": 0}

        with mock.patch.object(da, "run_restoration", side_effect=fake):
            imgs, met = da.restore_damage_aware(img, m, None, "sd_edge", {})
        self.assertEqual([c[1] for c in calls], sorted(da.TIERS))        # light -> heavy
        self.assertEqual(calls[0][0], "sd")                             # cracks skip the edge stage
        self.assertTrue(all(c[0] == "sd_edge" for c in calls[1:]))
        self.assertEqual(sum(c[2] for c in calls), int((np.asarray(m) > 127).sum()))
        self.assertIn("types", imgs); self.assertEqual(met["damage_summary"]["regions"], 3)

    def test_crack_classical_backend(self):
        img, m = scene()
        with mock.patch.object(da, "run_restoration", side_effect=lambda i, k, mo, c: ({"stage1": i}, {})) as rr:
            _, met = da.restore_damage_aware(img, m, None, "sd", {"crack_backend": "classical"})
        self.assertEqual(rr.call_count, 2)
        self.assertEqual(met["tier_calls"][0]["backend"], "classical")

    def test_end_to_end_classical_keeps_intact_pixels(self):
        img, m = scene()
        res = restore_mural(img, m, config_dict={"restore_mode": "classical", "damage_aware": True})
        o, r = np.asarray(img), np.asarray(res.restored_image)
        far = cv2.dilate(np.asarray(res.refined_mask), np.ones((41, 41), np.uint8)) == 0
        self.assertTrue(np.array_equal(o[far], r[far]))
        self.assertTrue(res.metrics["damage_aware"])
        self.assertIn("types", res.stages)


class TestMarginAndLpipsBox(unittest.TestCase):
    def test_dilated_crack_still_classified_as_crack_with_margin(self):
        m = np.zeros((512, 512), np.uint8)
        cv2.line(m, (50, 100), (450, 140), 255, 3)
        d = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))     # 4 px refine margin
        self.assertEqual([r.type for r in da.analyze_regions(d, margin_px=4)], ["crack"])

    def test_lpips_bbox_has_min_size(self):
        from evaluation.restoration_metrics import _bbox
        m = np.zeros((512, 512), bool); m[200:203, 300:302] = True
        y0, y1, x0, x1 = _bbox(m)
        self.assertTrue(y1 - y0 >= 64 and x1 - x0 >= 64)

class TestFullPipelineOrder(unittest.TestCase):
    def test_stages_run_in_order(self):
        import model.inference as inf
        img = Image.new("RGB", (256, 256), (150, 120, 90))
        m = Image.new("L", (256, 256), 0); m.paste(255, (80, 80, 160, 160))
        order = []

        def fake_da(image, mask, prob, mode, cfg):
            order.append(("damage_aware+stage1+stage2", mode))
            return {"stage1": Image.new("RGB", image.size, (10, 200, 10)),
                    "stage2": Image.new("RGB", image.size, (10, 10, 200))}, {"backend": "fake"}

        def fake_final(image, mask, cfg):
            order.append(("final_diffusion", None))
            return Image.new("RGB", image.size, (200, 10, 10)), mask, {"final_strength": 0.3}

        with mock.patch.object(inf, "restore_damage_aware", side_effect=fake_da), \
             mock.patch.object(inf, "final_diffusion", side_effect=fake_final):
            res = inf.restore_mural(img, m, config_dict={"restore_mode": "full"})
        self.assertEqual([o[0] for o in order], ["damage_aware+stage1+stage2", "final_diffusion"])
        self.assertEqual(order[0][1], "full")
        for k in ("stage1", "stage2", "final_diffusion", "final"):
            self.assertIn(k, res.stages)
        self.assertIn("post_stages", res.metrics)
        # intact pixels far from the damage stay bit-exact
        self.assertEqual(np.asarray(res.restored_image)[5, 5].tolist(), [150, 120, 90])


if __name__ == "__main__":
    unittest.main()
