"""Optional Docker checks: source and prepared caches survive --rm containers."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import uuid


@unittest.skipUnless(os.environ.get("PAITON_DOCKER_TESTS") == "1", "Set PAITON_DOCKER_TESTS=1 on the qualified Docker host")
class CachePersistenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        base = json.loads((cls.root / "container-images.json").read_text())["images"]["klein"]["reference"]
        cls.image = "paiton-flux2-cache-check:" + uuid.uuid4().hex
        recipe = f'FROM {base}\nENTRYPOINT ["python3"]\nCMD []\n'
        subprocess.run(["docker", "build", "-q", "-t", cls.image, "-"],
                       input=recipe, text=True, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        subprocess.run(["docker", "image", "rm", cls.image], check=True, capture_output=True)

    def check_cache(self, cache):
        marker = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as output:
            env = dict(os.environ, PAITON_IMAGE=self.image, PAITON_CACHE=cache, PAITON_OUTPUTS=output)
            files = ["/models/prepared-marker", "/models/cache/source-marker"]
            write = f"from pathlib import Path; [(Path(p).write_text({marker!r})) for p in {files!r}]"
            read = f"from pathlib import Path; assert all(Path(p).read_text()=={marker!r} for p in {files!r})"
            for code in (write, read):
                subprocess.run([str(self.root / "run.sh"), "-c", code], env=env,
                               check=True, capture_output=True, text=True)
        return marker

    def test_named_cache_survives_container_removal(self):
        prefix = "paiton-flux2-cache-test-" + uuid.uuid4().hex
        try:
            self.check_cache(prefix)
            volumes = json.loads(subprocess.check_output(["docker", "volume", "inspect", prefix, prefix + "-runtime"]))
            self.assertEqual({v["Name"] for v in volumes}, {prefix, prefix + "-runtime"})
        finally:
            subprocess.run(["docker", "volume", "rm", prefix, prefix + "-runtime"], check=True, capture_output=True)

    def test_bind_cache_is_visible_on_host_and_next_container(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = self.check_cache(directory)
            self.assertEqual((Path(directory) / "cache/source-marker").read_text(), marker)
            self.assertEqual((Path(directory) / "prepared-marker").read_text(), marker)


if __name__ == "__main__":
    unittest.main()
