import unittest
from PIL import Image
from paiton_image21.runtime import validate_request

class RequestLimits(unittest.TestCase):
    def test_image_edit_contract(self):
        image=Image.new("RGBA",(1024,1024))
        settings=validate_request("Make the teapot red",1024,1024,mode="edit",image=image)
        self.assertEqual(settings["num_inference_steps"],40)
        self.assertEqual(settings["true_cfg_scale"],1.0)
        self.assertTrue(settings["use_kv_cache"])
        with self.assertRaises(ValueError):
            validate_request("edit",2048,2048,mode="edit",image=image)

    def test_unqualified_requests_fail_before_gpu_work(self):
        invalid=[dict(prompt=""),dict(prompt="x"*513),dict(prompt="x",width=4096,height=4096),
                 dict(prompt="x",width=1024,height=2048),dict(prompt="x",steps=4),
                 dict(prompt="x",guidance=2.0),dict(prompt="x",seed=-1),dict(prompt="x",seed=True),
                 dict(prompt="x",mode="edit"),dict(prompt="x",mode="video"),dict(prompt=None)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                validate_request(**kwargs)

if __name__=="__main__":
    unittest.main()
