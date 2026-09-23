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

    def test_model_selection_before_and_after_command(self):
        for args in (["--model", "uncensored", "serve"], ["serve", "--model", "uncensored"]):
            with self.subTest(args=args), patch("paiton_image21.checkpoint.prepare", return_value=Path("/uncensored")) as prepare, patch.object(run.os, "execve", side_effect=Executed) as execute, patch.dict(run.os.environ, {}, clear=True):
                with self.assertRaises(Executed):
                    run.main(args)
                self.assertEqual(prepare.call_args.kwargs["model"], "uncensored")
                command=execute.call_args.args[1]
                self.assertEqual(command[command.index("--precision-profile")+1], "exact")
                self.assertIn("/uncensored", command)

    def test_original_profile_is_preserved(self):
        with patch("paiton_image21.checkpoint.prepare", return_value=Path("/original")) as prepare, patch.object(run.os, "execve", side_effect=Executed) as execute, patch.dict(run.os.environ, {}, clear=True):
            with self.assertRaises(Executed):
                run.main([])
            self.assertEqual(prepare.call_args.kwargs["model"], "original")
            command=execute.call_args.args[1]
            self.assertEqual(command[command.index("--precision-profile")+1], "schedule-int8")

    def test_explicit_profile_override_is_retained(self):
        with patch("paiton_image21.checkpoint.prepare", return_value=Path("/uncensored")), patch.object(run.os, "execve", side_effect=Executed) as execute:
            with self.assertRaises(Executed):
                run.main(["serve", "--model", "uncensored", "--precision-profile", "schedule-int8-11"])
            command=execute.call_args.args[1]
            self.assertEqual(command[command.index("--precision-profile")+1], "schedule-int8-11")


if __name__ == "__main__":
    unittest.main()
