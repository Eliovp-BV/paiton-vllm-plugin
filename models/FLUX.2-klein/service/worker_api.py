"""Local image API with an unloadable, persistent inference worker."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler,HTTPServer
from io import BytesIO
import json
import multiprocessing
import time


def worker(connection,engine):
    import torch
    if engine=='paiton':
        from paiton_image.pipeline import load_pipeline,generate
    else:
        from sdnq_tool.pipeline import load_pipeline,generate
    started=time.perf_counter()
    try:
        pipe=load_pipeline(engine)
        torch.cuda.synchronize()
        load_seconds=time.perf_counter()-started
    except Exception as error:
        connection.send({'error':str(error)});return
    first=True
    while True:
        request=connection.recv()
        try:
            torch.cuda.synchronize();started=time.perf_counter()
            image=generate(pipe,request['prompt'],request['seed']).images[0]
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            buffer=BytesIO();image.save(buffer,format='PNG',compress_level=1)
            connection.send({'image':base64.b64encode(buffer.getvalue()).decode(),
                'engine':engine,'seconds':elapsed,'first_generation':first,
                'load_seconds':load_seconds if first else 0,'seed':request['seed'],
                'resolution':[1024,1024],'steps':4})
            first=False
        except Exception as error:
            connection.send({'error':str(error)})


class InferenceWorker:
    def __init__(self,engine):
        self.engine=engine;self.process=None;self.connection=None

    def unload(self):
        if self.process is not None:
            self.process.terminate();self.process.join(30)
            if self.process.is_alive():
                self.process.kill();self.process.join()
            self.connection.close();self.process=None;self.connection=None

    def generate(self,request):
        if self.process is None or not self.process.is_alive():
            self.unload()
            context=multiprocessing.get_context('spawn')
            self.connection,child=context.Pipe()
            self.process=context.Process(target=worker,args=(child,self.engine))
            self.process.start();child.close()
        self.connection.send(request)
        deadline=time.monotonic()+1200
        while not self.connection.poll(1):
            if not self.process.is_alive():
                raise RuntimeError('The inference worker exited. Check the engine container log.')
            if time.monotonic()>deadline:
                self.unload();raise RuntimeError('Generation exceeded the 20-minute startup limit.')
        return self.connection.recv()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--engine',choices=['paiton','stock'],required=True)
    parser.add_argument('--port',type=int,default=7860);args=parser.parse_args()
    inference=InferenceWorker(args.engine)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,data,status=200):
            body=json.dumps(data).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path!='/health':self.reply({'error':'Not found'},404);return
            self.reply({'ready':True,'engine':args.engine,'worker_running':inference.process is not None and inference.process.is_alive()})
        def do_POST(self):
            if self.path=='/unload':
                inference.unload();self.reply({'unloaded':True});return
            if self.path!='/generate':self.reply({'error':'Not found'},404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=40000:raise ValueError('Enter a prompt of 1 to 8000 characters.')
                request=json.loads(self.rfile.read(length))
                prompt=request.get('prompt');seed=request.get('seed',42)
                if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>8000:
                    raise ValueError('Enter a prompt of 1 to 8000 characters.')
                if type(seed) is not int or not 0<=seed<2**64:raise ValueError('Seed must be an unsigned 64-bit integer.')
            except (ValueError,TypeError,AttributeError,UnicodeError) as error:
                self.reply({'error':str(error)},400);return
            try:
                response=inference.generate({'prompt':prompt,'seed':seed})
                self.reply(response,500 if 'error' in response else 200)
            except (RuntimeError,EOFError,BrokenPipeError) as error:
                inference.unload();self.reply({'error':str(error)},500)
        def log_message(self,*args):
            pass
    print(f'{args.engine} image API ready on port {args.port}',flush=True)
    try:HTTPServer(('0.0.0.0',args.port),Handler).serve_forever()
    finally:inference.unload()


if __name__=='__main__':main()
