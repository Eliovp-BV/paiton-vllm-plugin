import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paiton_runtime_compat.cache_namespace import configure


class CacheTests(unittest.TestCase):
    def test_spawn_reentry_and_code_change_do_not_nest_or_reuse_cache(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            root=Path(d);(root/'bootstrap.py').write_text('original')
            os.environ['VLLM_CACHE_ROOT']='/cache/explicit'
            manifest={'files':{}}
            first=configure(root,manifest); original=os.environ['VLLM_CACHE_ROOT']
            self.assertEqual(configure(root,manifest),first)
            self.assertEqual(os.environ['VLLM_CACHE_ROOT'],original)
            (root/'bootstrap.py').write_text('changed implementation')
            second=configure(root,manifest)
            self.assertNotEqual(first,second)
            self.assertEqual(os.environ['VLLM_CACHE_ROOT'],f'/cache/explicit/paiton-runtime/{second}')


if __name__=='__main__':unittest.main()
