## Gradio demo: VILA with TinyChat

We provide scripts for building your own gradio server to run VILA models with TinyChat. Please run the following commands to launch the server.

#### Launch a controller
```bash
python -m tinychat.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch gradio web server.
```bash
python -m tinychat.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --share --auto-pad-image-token
```
After launching this script, the web interface will be served on your machine and you can access it with a public URL (or localhost URL).

#### Launch a model worker

```bash
python -m tinychat.serve.model_worker_new --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path <path-to-fp16-hf-model> --quant-path <path-to-awq-checkpoint>
# Please change tinychat.serve.model_worker_new to tinychat.serve.model_worker if you want to serve VILA rather than VILA-1.5
```

Note: You can launch multiple model workers onto the same web server. And please remember to specify different ports for each model worker.

### Acknowlegement

This demo is inspired by [LLaVA](https://github.com/haotian-liu/LLaVA). We thank LLaVA for providing an elegant way to build the Gradio Web UI.
