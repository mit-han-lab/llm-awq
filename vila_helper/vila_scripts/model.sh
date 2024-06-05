python -m tinychat.serve.model_worker_new --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path $1 --quant-path $1/llm/*.pt
