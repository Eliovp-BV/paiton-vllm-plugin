"""Install the inert process-start hook next to site-packages, not globally."""

from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        Path(self.build_lib, "000_paiton_explicit.pth").write_text(
            "import paiton_vllm_plugin.startup\n"
        )


setup(cmdclass={"build_py": BuildPy})
