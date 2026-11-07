"""Print the numbers needed for the model table (layers, heads, context, vocab, generation defaults).
Run from the repo root:  python -m src.show_config"""
from transformers import AutoConfig, GenerationConfig

for name in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
    c = AutoConfig.from_pretrained(name)
    print(f"\n=== {name}")
    for k in ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
              "max_position_embeddings", "vocab_size", "tie_word_embeddings"]:
        print(f"  {k}: {getattr(c, k, None)}")
    g = GenerationConfig.from_pretrained(name)
    print("  generation_config defaults -> repetition_penalty:", g.repetition_penalty,
          "| temperature:", g.temperature, "| top_p:", g.top_p, "| top_k:", g.top_k, "| do_sample:", g.do_sample)