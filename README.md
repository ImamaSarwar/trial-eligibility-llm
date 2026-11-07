# Clinical-trial eligibility screening with a LoRA-tuned Qwen2.5 (educational prototype)

Educational screening prototype - **not** a clinical enrollment decision.

## Layout
- `src/data_prep.py` split (seed 42, 37/8/8 patients), manifest, leakage checks (`data/split_manifest.json`)
- `src/train_lora.py`, `src/evaluate.py`, `src/metrics.py` fine-tuning and evaluation (Tasks 1-2)
- `src/kv_decode.py` hand-written no-cache / KV-cache greedy decoding (Task 3)
- `src/attention.py`, `src/bench_attention.py` reference + tiled online-softmax attention (Task 4)
- `src/speculative.py`, `src/benchmark.py` greedy speculative decoding and benchmarks (Task 5)
- `app/app.py` Gradio browser app (Task 6); `tests/` pytest suite (runs on CPU in seconds)

## Setup (Python 3.12)
    python -m venv .venv && .venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
    pip install -r requirements.txt
    python src/data_prep.py        # writes data/processed/*.jsonl and data/split_manifest.json
    python -m pytest -q            # attention / KV-cache / speculative / metrics tests

## Train + evaluate (Colab T4)
    python -m src.train_lora --out adapters/qwen15b_lora --push_to_hub <user>/<repo>
    python -m src.evaluate --split val --tag base
    python -m src.evaluate --split val --tag ft --adapter adapters/qwen15b_lora/best
    python -m src.evaluate --split test --tag ft --adapter adapters/qwen15b_lora/best --confirm_test   # once
    python -m src.bench_attention
    python -m src.benchmark --tag t4_fp16 --adapter adapters/qwen15b_lora/best

## Run the app
    set ADAPTER=<user>/<repo>      (PowerShell: $env:ADAPTER="<user>/<repo>")
    python app/app.py              # open http://127.0.0.1:7860

## AI-use statement
AI assistants (Claude) were used to help draft code; all code was reviewed, tested, and is understood by the author.
