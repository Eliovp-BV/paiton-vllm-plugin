"""Exercise engine switching and IMAGE conversion over a real local HTTP socket."""
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from PIL import Image
import torch

path=Path(__file__).resolve().parents[1]/'comfyui/custom_nodes/paiton_flux2/__init__.py'
spec=importlib.util.spec_from_file_location('paiton_comfy_node',path)
node=importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class ComfyBridgeTest(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.error=False
        owner=self
        buffer=BytesIO()
        Image.new('RGB',(3,2),(255,128,0)).save(buffer,format='PNG')
        encoded=base64.b64encode(buffer.getvalue()).decode()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.calls.append((self.path,data))
                result={'error':'Model cache is missing'} if owner.error else {'image':encoded}
                body=json.dumps(result).encode()
                self.send_response(500 if owner.error else 200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
            def log_message(self,*args):pass
        self.server=HTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.original=node._endpoints
        address=f'http://127.0.0.1:{self.server.server_port}'
        node._endpoints={'Paiton':address+'/paiton','Stock (Diffusers)':address+'/stock'}

    def tearDown(self):
        node._endpoints=self.original
        self.server.shutdown();self.thread.join();self.server.server_close()

    def test_switch_unloads_other_engine_before_generation(self):
        for engine,active,other in [('Paiton','paiton','stock'),('Stock (Diffusers)','stock','paiton')]:
            self.calls.clear()
            image,=node.PaitonFlux2().generate(engine,'A teal cup',42)
            self.assertEqual(self.calls,[(f'/{other}/unload',{}),
                (f'/{active}/generate',{'prompt':'A teal cup','seed':42})])
            self.assertEqual(tuple(image.shape),(1,2,3,3))
            self.assertEqual(image.device.type,'cpu')
            self.assertEqual(image.dtype,torch.float32)
            torch.testing.assert_close(image[0,0,0],torch.tensor([1,128/255,0]))

    def test_engine_error_reaches_the_user(self):
        self.error=True
        with self.assertRaisesRegex(RuntimeError,'Model cache is missing'):
            node.PaitonFlux2().generate('Paiton','A teal cup',42)
        self.assertEqual(len(self.calls),1)

    def test_invalid_engine_does_not_send_a_request(self):
        with self.assertRaises(ValueError):
            node.PaitonFlux2().generate('invalid','A teal cup',42)
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
