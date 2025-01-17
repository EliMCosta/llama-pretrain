#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys
import tracemalloc
import psutil
import torch

from dotenv import load_dotenv
from transformers import (
    PreTrainedTokenizerFast,
    GenerationConfig
)

from train_llama import LlamaForCausalLM

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run inference on Raven model with generation_config.json support.")
    parser.add_argument("--model_path", type=str, default="./out/final_model",
                        help="Path to directory with model files (config.json, generation_config.json, tokenizer, etc.).")
    parser.add_argument("--prompt", type=str, default="The city",
                        help="Prompt text to feed the model.")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Override for max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Override for temperature.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Override for top_p sampling.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Override for top_k sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.5,
                        help="Override for repetition_penalty.")
    parser.add_argument("--num_beams", type=int, default=None,
                        help="Override for num_beams.")
    parser.add_argument("--num_cpus", type=int, default=0,
                        help="If > 0, run on CPU with this many threads; otherwise try GPU.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Preferred device to run inference on (e.g. 'cuda', 'cpu'). Ignored if --num_cpus > 0.")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Decide on device
    # --------------------------------------------------------
    # 1) If user specified --num_cpus > 0, force CPU usage
    #    and set the CPU threads.
    if args.num_cpus > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.num_cpus)
        torch.set_num_threads(args.num_cpus)
        device = torch.device("cpu")
        print(f"[INFO] Using CPU with {args.num_cpus} threads.")
    else:
        # 2) Otherwise, check if CUDA is available and
        #    optionally override with --device
        if torch.cuda.is_available() and args.device == "cuda":
            device = torch.device("cuda")
            print("[INFO] Using CUDA.")
        else:
            device = torch.device("cpu")
            print("[INFO] Using CPU (no GPU available or device overridden).")

    # --------------------------------------------------------
    # 1) Load the trained Raven model from disk
    # --------------------------------------------------------
    try:
        model = LlamaForCausalLM.from_pretrained(args.model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"[INFO] Loaded trained model from: {args.model_path}")
    except Exception as e:
        print(f"[ERROR] Could not load LlamaForCausalLM.from_pretrained({args.model_path}). Error: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # 2) Load the tokenizer
    # --------------------------------------------------------
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
        print("[INFO] Loaded tokenizer from the model folder.")
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer from {args.model_path}. Error: {e}")
        sys.exit(1)

    # If tokenizer's vocab != model config, adjust:
    if len(tokenizer) != model.config.vocab_size:
        print(f"[INFO] Adjusting model's vocab_size {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # --------------------------------------------------------
    # 3) Load generation_config.json + apply CLI overrides
    # --------------------------------------------------------
    try:
        base_generation_config = GenerationConfig.from_pretrained(args.model_path)
        print("[INFO] Loaded generation_config.json from the model folder.")
    except Exception:
        base_generation_config = GenerationConfig()
        print("[WARNING] No valid generation_config.json found. Using default GenerationConfig.")

    # Decide whether to sample
    do_sample = base_generation_config.do_sample
    # If user changes temperature, top_p, or top_k, we want sampling
    if (args.temperature is not None and args.temperature != 1.0) \
       or (args.top_p is not None and args.top_p != 1.0) \
       or (args.top_k is not None and args.top_k > 0):
        do_sample = True

    # Build final GenerationConfig
    gen_config = GenerationConfig(**base_generation_config.to_dict())
    if args.max_new_tokens is not None:
        gen_config.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        gen_config.temperature = args.temperature
    if args.top_p is not None:
        gen_config.top_p = args.top_p
    if args.top_k is not None:
        gen_config.top_k = args.top_k
    if args.repetition_penalty is not None:
        gen_config.repetition_penalty = args.repetition_penalty
    if args.num_beams is not None:
        gen_config.num_beams = args.num_beams
    gen_config.do_sample = do_sample

    # If user set num_beams > 1 and also do_sample = True, that's
    # usually contradictory (beam search vs. random sampling).
    # We'll just let Transformers handle it, but let's warn:
    if gen_config.num_beams is not None and gen_config.num_beams > 1 and do_sample:
        print("[WARNING] Both num_beams>1 and do_sample=True can conflict. Proceeding anyway...")

    # --------------------------------------------------------
    # 4) Ensure pad_token_id is consistent
    # --------------------------------------------------------
    if gen_config.pad_token_id is not None:
        if (tokenizer.pad_token_id is None) or (tokenizer.pad_token_id != gen_config.pad_token_id):
            tokenizer.pad_token_id = gen_config.pad_token_id
            print(f"[INFO] Setting tokenizer.pad_token_id = {gen_config.pad_token_id}")
    if model.config.pad_token_id is None:
        model.config.pad_token_id = gen_config.pad_token_id

    # --------------------------------------------------------
    # 5) Encode the prompt
    # --------------------------------------------------------
    prompt_text = args.prompt
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    # --------------------------------------------------------
    # 6) Track memory/time usage
    # --------------------------------------------------------
    tracemalloc.start()  # Track CPU memory usage snapshots
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1_048_576  # MB
    start_time = time.time()

    # --------------------------------------------------------
    # 7) Generate text
    # --------------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            generation_config=gen_config,
        )

    end_time = time.time()
    current_mem_usage, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1_048_576  # MB

    # For GPU, record peak memory
    if device.type == "cuda":
        peak_memory_allocated = torch.cuda.max_memory_allocated(device) / 1_048_576
    else:
        # Approx. difference on CPU
        peak_memory_allocated = max(0.0, end_mem_usage - start_mem_usage)

    # --------------------------------------------------------
    # 8) Decode and print results
    # --------------------------------------------------------
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    total_time = end_time - start_time
    tokens_generated = len(output_ids[0]) - input_ids.shape[1]
    tokens_per_second = tokens_generated / total_time if total_time > 0 else float("inf")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    print("\n=== GENERATED TEXT ===")
    print(generated_text)
    print(
        f"\n=== Generation Metrics ===\n"
        f"Prompt: {prompt_text}\n"
        f"Total tokens generated: {tokens_generated}\n"
        f"Time taken: {total_time:.2f} seconds\n"
        f"Tokens per second: {tokens_per_second:.2f}\n"
        f"Hidden size: {hidden_size}, Layers: {num_layers}\n"
        f"Total parameters: {total_params:,}\n"
        f"Trainable parameters: {trainable_params:,}\n"
        f"Peak memory usage: {peak_memory_allocated:.2f} MB\n"
        f"Device: {device}\n"
    )

if __name__ == "__main__":
    main()
