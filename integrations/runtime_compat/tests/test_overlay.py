import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_runtime_compat.bootstrap import OverlayFinder


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class OverlayTests(unittest.TestCase):
    def test_package_siblings_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pkg=root/'fixture_runtime_pkg';pkg.mkdir()
            original=pkg/'__init__.py';original.write_text('value = 1\n')
            (pkg/'child.py').write_text('value = 7\n')
            overlay=root/'overlay.py';overlay.write_text('value = 2\ndef selected():\n    return 2\n')
            manifest={'files':{str(original):{'overlay':'overlay.py','original_sha256':digest(original),'overlay_sha256':digest(overlay)}}}
            finder=OverlayFinder(manifest,root)
            sys.path.insert(0,str(root));sys.meta_path.insert(0,finder)
            try:
                mod=importlib.import_module('fixture_runtime_pkg')
                child=importlib.import_module('fixture_runtime_pkg.child')
                self.assertEqual((mod.value,child.value),(2,7))
                self.assertEqual(mod.__path__,[str(pkg)])
                self.assertEqual(mod.__file__,str(original))
                self.assertEqual(mod.selected.__code__.co_filename,str(original))
                self.assertIn('return 2',inspect.getsource(mod.selected))
                self.assertEqual(original.read_text(),'value = 1\n')
                self.assertEqual(finder.loaded,{'fixture_runtime_pkg'})
            finally:
                sys.path.remove(str(root));sys.meta_path.remove(finder)
                for name in ('fixture_runtime_pkg.child','fixture_runtime_pkg'):sys.modules.pop(name,None)

    def test_source_and_payload_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'fixture_runtime_mod.py';original.write_text('value=1\n')
            overlay=root/'overlay.py';overlay.write_text('value=2\n')
            entry={'overlay':'overlay.py','original_sha256':digest(original),'overlay_sha256':digest(overlay)}
            finder=OverlayFinder({'files':{str(original):entry}},root)
            original.write_text('value=3\n')
            with self.assertRaisesRegex(ImportError,'original-source'):
                finder.find_spec('fixture_runtime_mod',[str(root)])
            entry['original_sha256']=digest(original);overlay.write_text('value=4\n')
            with self.assertRaisesRegex(ImportError,'payload mismatch'):
                finder.find_spec('fixture_runtime_mod',[str(root)])

if __name__=='__main__':unittest.main()
