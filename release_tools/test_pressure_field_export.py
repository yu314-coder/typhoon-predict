"""Regression checks for the capture-only released 1.2 pressure export."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/trackformer_1_2_field"))
from predict import export_pressure_fields
from plot_pressure import render_pressure_map


class PressureFieldExportTest(unittest.TestCase):
    def setUp(self):
        self.contract = {"normalization": {"std": [40], "mean": [1000]}}
        lat, lon = torch.meshgrid(torch.linspace(60, 0, 25), torch.linspace(100, 180, 33), indexing="ij")
        rlat, rlon = torch.meshgrid(torch.linspace(30, 10, 121), torch.linspace(120, 140, 121), indexing="ij")
        static = lambda y, x: torch.stack((y/90, x/180-1, torch.zeros_like(y), torch.zeros_like(y)))[None]
        self.inputs = {"global_static": static(lat, lon), "regional_static": static(rlat, rlon),
                       "center": torch.tensor([[20., 130.]])}
        self.outputs = []
        for i in range(20):
            y, x = torch.meshgrid(torch.linspace(25, 15, 65), torch.linspace(125+i*.5, 135+i*.5, 65), indexing="ij")
            core = -torch.exp(-((x-(130+i*.5))**2+(y-20)**2)/3)[None, None]
            mask = torch.ones_like(core, dtype=torch.bool)
            mask[:, :, :, 0] = False
            self.outputs.append({"core": core, "core_lat": y[None], "core_lon": x[None], "core_valid": mask,
                "regional_valid": torch.ones((1,1,121,121),dtype=torch.bool), "track_valid": torch.tensor([True]),
                "pressure": torch.tensor([800.])})
        self.issue = 1536624000 * 10**9

    def test_core_units_geography_and_every_valid_time(self):
        result = export_pressure_fields(self.outputs, self.inputs, self.contract, self.issue)
        self.assertEqual(result["core_mslp_hpa"].shape, (20,65,65))
        self.assertEqual(result["core_latitude_deg"].shape, (20,65,65))
        self.assertEqual(result["core_longitude_deg"].shape, (20,65,65))
        self.assertEqual(result["core_valid"].dtype, np.dtype(bool))
        self.assertEqual(result["regional_valid"].shape, (20,121,121))
        np.testing.assert_array_equal(result["valid_time_ns"], self.issue+np.arange(6,121,6,dtype=np.int64)*3600*10**9)
        self.assertEqual(int(result["member_count"]), 1)
        self.assertEqual(float(result["core_information_spacing_km"]), 20)
        self.assertAlmostEqual(float(result["core_mslp_hpa"][0,32,32]),960)
        np.testing.assert_array_equal(result["core_longitude_deg"][19], self.outputs[19]["core_lon"][0].numpy())
        self.assertFalse(result["core_valid"][:,:,0].any())

    def test_capture_preserves_tensors_and_never_inserts_scalar_pressure(self):
        before = [{k:v.clone() for k,v in o.items()} for o in self.outputs]
        result = export_pressure_fields(self.outputs, self.inputs, self.contract, self.issue)
        self.assertGreater(float(result["core_mslp_hpa"].min()), float(self.outputs[0]["pressure"][0]))
        for old, new in zip(before, self.outputs):
            for key in old:
                self.assertTrue(torch.equal(old[key], new[key]), key)

    def test_a_moving_core_can_leave_the_original_fixed_patch(self):
        result = export_pressure_fields(self.outputs, self.inputs, self.contract, self.issue)
        fixed_east = float(result["regional_longitude_deg"].max())
        self.assertGreater(float(result["core_longitude_deg"][-1].max()), fixed_east)
        np.testing.assert_array_equal(result["regional_longitude_deg"], (self.inputs["regional_static"][0,1].numpy()+1)*180)

    def test_incomplete_rollout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "twenty"):
            export_pressure_fields(self.outputs[:-1], self.inputs, self.contract, self.issue)

    def plot_data(self):
        result = export_pressure_fields(self.outputs, self.inputs, self.contract, self.issue)
        result.update(lead_hours=np.arange(6,121,6), basin_mslp_hpa=np.stack([
            1005+self.inputs["global_static"][0,0].numpy()*10+i*.1 for i in range(20)]),
            central_pressure_hpa=np.full(20,960),
            track_lat_lon=np.column_stack((np.full(20,20),130+np.arange(20)*.5)))
        return result

    def test_renderer_accepts_real_geography_and_distinct_leads(self):
        with tempfile.TemporaryDirectory(prefix="pressure-export-test-") as folder:
            first, last = Path(folder)/"first.png", Path(folder)/"last.png"
            data = self.plot_data()
            render_pressure_map(data, first, 6, 4)
            render_pressure_map(data, last, 120, 4)
            self.assertGreater(first.stat().st_size, 10000)
            self.assertNotEqual(hashlib.sha256(first.read_bytes()).digest(), hashlib.sha256(last.read_bytes()).digest())

    def test_missing_core_is_masked_not_filled_from_scalar(self):
        data = self.plot_data()
        data["core_valid"][:] = False
        original = data["core_mslp_hpa"].copy()
        with tempfile.TemporaryDirectory(prefix="pressure-export-test-") as folder:
            render_pressure_map(data, Path(folder)/"masked.png", 120)
        np.testing.assert_array_equal(data["core_mslp_hpa"], original)

    def test_invalid_contour_interval_and_unregistered_ensemble_are_rejected(self):
        for interval in (0,-1,np.nan):
            with self.assertRaisesRegex(ValueError, "interval"):
                render_pressure_map(self.plot_data(), "unused.png", interval=interval)
        data = self.plot_data()
        data["member_count"] = np.array(50)
        with self.assertRaisesRegex(ValueError, "one-member"):
            render_pressure_map(data, "unused.png")

    def test_released_neural_source_hashes_are_unchanged(self):
        directory = ROOT/"models/trackformer_1_2_field"
        manifest = json.loads((directory/"manifest.json").read_text())
        for name, expected in manifest["source_module_sha256"].items():
            self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
