import json
import os

def parse_time(val):
    if val is None:
        return 0.0
    if isinstance(val, list):
        valid_nums = [x for x in val if isinstance(x, (int, float))]
        return float(sum(valid_nums)) if valid_nums else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def generate_benchmark_table():
    results_dir = "results"
    target_file = os.path.join(results_dir, "benchmark_t4_fp16_v2.json")
    
    if not os.path.exists(target_file):
        print(f"Error: '{target_file}' not found.")
        return

    with open(target_file, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No 'results' key found in the JSON file.")
        return

    for r in results:
        prompt_len = r.get("prompt_len", r.get("prompt_length", "-"))
        
        # Extended search for generation length field name
        gen_len = r.get("gen_len", r.get("gen_tokens", r.get("max_new_tokens", r.get("target_gen_len", r.get("new_tokens", "-")))))
        config = f"{prompt_len}/{gen_len}"
        
        mode = r.get("mode", "unknown")
        
        raw_prefill = r.get("prefill_time", r.get("prefill_s", r.get("time_to_first_token", 0.0)))
        raw_decode = r.get("decode_time", r.get("decode_s", r.get("generation_time", r.get("decode_times", 0.0))))
        
        prefill_time = parse_time(raw_prefill)
        decode_time = parse_time(raw_decode)
        
        print(f"{config} & {mode} & {prefill_time:.2f} & {decode_time:.2f} \\\\")

if __name__ == "__main__":
    generate_benchmark_table()