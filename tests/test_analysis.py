import unittest

import numpy as np
import pandas as pd

from src.analysis import (synthetic_landmarks, head_orientation,
                          count_spins, analyse_sessions)


class HeadDirectionTests(unittest.TestCase):
    def test_synthetic_sessions_have_expected_groups(self):
        pose = synthetic_landmarks()
        self.assertEqual(set(pose.condition), {"baseline", "stimulation"})
        self.assertEqual(len(pose), 1200)

    def test_angle_unwrapping(self):
        df = synthetic_landmarks()
        subset = df[df.condition == "stimulation"]
        out = head_orientation(subset)
        angle = out.unwrapped_deg.to_numpy()
        self.assertGreater(angle[-1] - angle[0], 1000)
        self.assertLess(np.max(np.abs(np.diff(angle))), 20)

    def test_spins_detected(self):
        _, summary = analyse_sessions(synthetic_landmarks())
        base = summary.set_index("condition").loc["baseline"]
        stim = summary.set_index("condition").loc["stimulation"]
        self.assertEqual(int(base.total_spins), 0)
        self.assertGreaterEqual(int(stim.counterclockwise_spins), 2)
        self.assertLessEqual(int(stim.counterclockwise_spins), 3)

    def test_clockwise_and_anticlockwise_bouts(self):
        t = np.linspace(0, 30, 601)
        ccw_angle = np.linspace(0, 750, 601)
        cw_angle = np.linspace(0, -750, 601)
        ccw = count_spins(ccw_angle, t)
        cw = count_spins(cw_angle, t)
        self.assertEqual(ccw["counterclockwise_spins"], 2)
        self.assertEqual(ccw["clockwise_spins"], 0)
        self.assertEqual(cw["clockwise_spins"], 2)
        self.assertEqual(cw["counterclockwise_spins"], 0)

    def test_low_confidence_is_interpolated(self):
        pose = synthetic_landmarks()
        pose.loc[pose.index[:2], "likelihood_nose"] = 0.2
        trace = head_orientation(pose[pose.condition == "baseline"])
        self.assertTrue(np.isfinite(trace.unwrapped_deg.to_numpy()).all())

    def test_missing_columns_rejected(self):
        with self.assertRaises(ValueError):
            analyse_sessions(pd.DataFrame({"time_s": [1,2,3]}))


if __name__ == "__main__":
    unittest.main()
