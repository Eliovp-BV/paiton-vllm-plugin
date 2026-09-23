"""The HTTP API identifies and routes only the loaded checkpoint; CPU-only."""
import io
import json
import unittest
from unittest.mock import Mock,patch
from paiton_image21 import server


class ModelIdentity(unittest.TestCase):
    def handler(self,model):
        engine=Mock(model_id=model,model_variant='uncensored' if model.endswith('uncensored') else 'original')
        engine.checkpoint_sha256='a'*64
        engine.generate.return_value=(None,b'png',{'model':model})
        with patch.object(server,'ThreadingHTTPServer') as http:
            server.serve(engine,'127.0.0.1',0)
            cls=http.call_args.args[1]
        handler=cls.__new__(cls)
        handler.send_json=Mock()
        return handler,engine

    def test_list_identifies_loaded_variant(self):
        for model in ('paiton-image-2.1','paiton-image-2.1-uncensored'):
            h,_=self.handler(model);h.path='/v1/models';h.do_GET()
            self.assertEqual(h.send_json.call_args.args[1]['data'][0]['id'],model)

    def test_wrong_model_is_rejected_before_generation(self):
        h,e=self.handler('paiton-image-2.1-uncensored')
        h.path='/v1/images/generations';body=json.dumps({'model':'paiton-image-2.1','prompt':'A teapot'}).encode()
        h.headers={'Content-Length':str(len(body))};h.rfile=io.BytesIO(body);h.do_POST()
        self.assertEqual(h.send_json.call_args.args[0],400);e.generate.assert_not_called()

    def test_omitted_model_uses_loaded_checkpoint(self):
        h,e=self.handler('paiton-image-2.1-uncensored')
        h.path='/v1/images/generations';body=json.dumps({'prompt':'A teapot'}).encode()
        h.headers={'Content-Length':str(len(body))};h.rfile=io.BytesIO(body);h.do_POST()
        self.assertEqual(h.send_json.call_args.args[0],200);e.generate.assert_called_once()
