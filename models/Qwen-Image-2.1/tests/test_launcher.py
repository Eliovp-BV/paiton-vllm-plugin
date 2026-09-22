"""Preparation and fallback options must survive the public launch helpers."""

from pathlib import Path
import unittest
from unittest.mock import patch

import run


class Executed(Exception):
    pass


class Launcher(unittest.TestCase):
    def test_offline_fallback_before_and_after_command(self):
        common = ["generate", "--prompt", "A blue teapot", "--output", "/unused/new-image.png"]
        for args in (["--offline", "--no-native-fusions", *common],
                     [*common, "--offline", "--no-native-fusions"]):
            with self.subTest(args=args), patch("paiton_image21.checkpoint.prepare", return_value=Path("/verified")) as prepare, patch.object(run.os, "execve", side_effect=Executed) as execute:
                with self.assertRaises(Executed):
                    run.main(args)
                self.assertTrue(prepare.call_args.args[2])
                command = execute.call_args.args[1]
                self.assertNotIn("--native-fusions", command)
                self.assertIn("/verified", command)

    def test_default_launcher_enables_optimized_regions(self):
        with patch("paiton_image21.checkpoint.prepare", return_value=Path("/verified")), patch.object(run.os, "execve", side_effect=Executed) as execute:
            with self.assertRaises(Executed):
                run.main([])
            self.assertIn("--native-fusions", execute.call_args.args[1])

    def test_download_only_does_not_start_engine(self):
        with patch("paiton_image21.checkpoint.prepare", return_value=Path("/verified")), patch.object(run.os, "execve", side_effect=AssertionError("must not start")):
            self.assertEqual(run.main(["serve", "--download-only"]), 0)


if __name__ == "__main__":
    unittest.main()
