# train_v5_standard_llama2.py
# -*- coding: utf-8 -*-
"""
Usage:
  Start training from scratch (no preprocessed dataset):
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py

  Use preprocessed dataset:
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py --preprocessed_data_dir ./preprocessed_data

  Resume training from a checkpoint:
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py --resume_from_checkpoint ./out/checkpoint-1000

  Continue training from a local model:
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py --model_path ./local_model

  Specify a tokenizer (local path or Hugging Face model):
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py --tokenizer_path ./tokenizer --datasets abideen/Cosmopedia-100k-pretrain
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py --tokenizer_path NousResearch/Hermes-3-Llama-3.1-8B

  **Mixing Multiple Datasets** (if not using preprocessed_data_dir):
    torchrun --nproc_per_node=2 train_v5_standard_llama2.py --datasets abideen/Cosmopedia-100k-pretrain wikimedia/wikipedia:20231101.en
"""

import os

# ===========================
# 🛠 Environment Configuration
# ===========================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TORCH_PROFILER"] = "0"

import argparse
import torch
import torch.distributed as dist
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, concatenate_datasets, load_from_disk
from huggingface_hub import login
from dotenv import load_dotenv
import multiprocessing

# Disable optimize_ddp for TorchDynamo
import torch._dynamo
torch._dynamo.config.optimize_ddp = False

num_cpus = multiprocessing.cpu_count()
torch.backends.cudnn.benchmark = True

load_dotenv()

# ===========================
# Model & Training Settings
# ===========================
HEADS = 4
DIMENSIONS = 256
LAYERS = 4
INTERMEDIATE_SIZE = 1024
CONTEXT_LENGTH = 1024
TRAINING_SEQUENCE_LENGTH = 512
NEW_MODEL = "Llama_Extended_v1"
HUGGINGFACE_ID = os.getenv('HF_USERNAME')

BATCH_SIZE = 350
LEARNING_RATE = 5e-4
EPOCHS = 10

def main_print(*args, **kwargs):
    """
    Prints only on rank 0 to avoid multi-process spam.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    # Initialize distributed if more than one GPU
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl', init_method='env://')

    parser = argparse.ArgumentParser(description="Train the Llama model.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to a checkpoint directory to resume training from')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a local model to continue training from')
    parser.add_argument('--tokenizer_path', type=str, default="NousResearch/Hermes-3-Llama-3.1-8B",
                        help='Path or name of the tokenizer to use')
    # Use preprocessed data if provided
    parser.add_argument('--preprocessed_data_dir', type=str, default=None,
                        help='Path to the directory with preprocessed data (from pretrain-datapreprocess.py)')
    # Original datasets argument (used only if not using preprocessed_data_dir)
    parser.add_argument('--datasets', type=str, nargs='+', default=["BEE-spoke-data/FineMeme-100k"],
                        help='List of datasets to use; only effective if --preprocessed_data_dir is not provided.')
    
    args = parser.parse_args()

    # Login to Hugging Face
    main_print("Authenticating with Hugging Face Hub...")
    login(token=os.getenv('HF_TOKEN'))

    # -----------------------
    # Load / Prepare Tokenizer
    # -----------------------
    if args.tokenizer_path is not None:
        if os.path.isdir(args.tokenizer_path) or os.path.isfile(args.tokenizer_path):
            main_print(f"Loading tokenizer from local path: {args.tokenizer_path}...")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                args.tokenizer_path,
                use_fast=True,
            )
        else:
            main_print(f"Loading tokenizer from HF model: {args.tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                use_fast=True,
            )
    else:
        main_print("Loading default tokenizer from tokenizer/tokenizer.json...")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="tokenizer/tokenizer.json",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            mask_token="<mask>",
        )

    # Add missing special tokens
    special_tokens_dict = {}
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        special_tokens_dict['pad_token'] = '<pad>'
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        special_tokens_dict['mask_token'] = '<mask>'
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        special_tokens_dict['eos_token'] = '</s>'
    if tokenizer.bos_token is None or tokenizer.bos_token_id is None:
        special_tokens_dict['bos_token'] = '<s>'
    if tokenizer.unk_token is None or tokenizer.unk_token_id is None:
        special_tokens_dict['unk_token'] = '<unk>'
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
        main_print(f"Added special tokens: {special_tokens_dict}")

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.eos_token = tokenizer.eos_token or '</s>'
    tokenizer.bos_token = tokenizer.bos_token or '<s>'
    tokenizer.mask_token = tokenizer.mask_token or '<mask>'
    tokenizer.unk_token = tokenizer.unk_token or '<unk>'

    tokenizer_vocab_size = len(tokenizer)
    main_print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

    # Print special token IDs
    main_print(f"Special tokens:")
    main_print(f"  pad_token_id: {tokenizer.pad_token_id}")
    main_print(f"  eos_token_id: {tokenizer.eos_token_id}")
    main_print(f"  bos_token_id: {tokenizer.bos_token_id}")
    main_print(f"  unk_token_id: {tokenizer.unk_token_id}")

    assert tokenizer.pad_token_id < tokenizer_vocab_size
    assert tokenizer.unk_token_id < tokenizer_vocab_size
    assert tokenizer.pad_token_id != tokenizer.unk_token_id

    # -----------------------
    # Load Preprocessed or Raw Dataset
    # -----------------------
    if args.preprocessed_data_dir:
        # 1) Load from preprocessed data
        main_print(f"Loading preprocessed dataset from: {args.preprocessed_data_dir}")
        data = load_from_disk(args.preprocessed_data_dir)
        
        # We assume that the data has ["train"] and optionally ["validation"] splits
        train_dataset = data["train"]
        if "validation" in data:
            eval_dataset = data["validation"]
            main_print(f"Loaded train split of size {len(train_dataset)}; validation split of size {len(eval_dataset)}")
        else:
            eval_dataset = None
            main_print(f"Loaded train split of size {len(train_dataset)}; no validation split found.")
        
        # The dataset presumably already has 'input_ids' columns, etc.
        # We'll let the data_collator handle the final creation of labels.
        tokenized_data = train_dataset  # for naming consistency
        total_tokens = len(train_dataset) * (train_dataset[0]['input_ids'].__len__())
        main_print(f"Token count estimation (train): {total_tokens:_}")
    else:
        # 2) If no preprocessed data dir is given, we fallback to loading from raw datasets
        main_print("No preprocessed_data_dir provided; loading raw datasets from Hugging Face or local files...")
        datasets_list = []
        for ds in args.datasets:
            if ':' in ds:
                dataset_name, data_dir = ds.split(':', 1)
            else:
                dataset_name = ds
                data_dir = None
            main_print(f"Loading dataset: {dataset_name}" + (f" with subfolder: {data_dir}" if data_dir else ""))
            dataset = load_dataset(dataset_name, data_dir=data_dir, split='train')
            datasets_list.append(dataset)

        data = concatenate_datasets(datasets_list)
        main_print(f"Total number of examples after concatenation: {len(data)}")

        def tokenize(element):
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=TRAINING_SEQUENCE_LENGTH,
                return_overflowing_tokens=True,
                padding='max_length',
            )
            input_ids = outputs["input_ids"]
            attention_masks = outputs["attention_mask"]
            labels = input_ids.copy()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
            }

        main_print("Tokenizing dataset...")
        tokenized_data = data.map(
            tokenize,
            batched=True,
            remove_columns=data.column_names,
            num_proc=max(1, num_cpus - 2)
        )

        total_tokens = len(tokenized_data) * TRAINING_SEQUENCE_LENGTH
        main_print(f"Training on {total_tokens:_} tokens")

        # No separate validation logic here, unless you choose to do a split

        train_dataset = tokenized_data
        eval_dataset = None  # Or do your own split if you want

    # -----------------------
    # Load / Initialize Model
    # -----------------------
    if args.model_path:
        main_print(f"Loading model from {args.model_path}...")
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True
        )
        # Resize token embeddings if needed
        if tokenizer_vocab_size != model.config.vocab_size:
            main_print("Resizing model embeddings to match tokenizer vocab size...")
            model.resize_token_embeddings(tokenizer_vocab_size)
    else:
        main_print("Initializing model from scratch using LlamaConfig...")
        config = LlamaConfig(
            vocab_size=tokenizer_vocab_size,
            max_position_embeddings=CONTEXT_LENGTH,
            hidden_size=DIMENSIONS,
            num_attention_heads=HEADS,
            num_hidden_layers=LAYERS,
            intermediate_size=INTERMEDIATE_SIZE,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            use_cache=True,
            tie_word_embeddings=True,
        )

        model = LlamaForCausalLM(config)

        if tokenizer_vocab_size != model.config.vocab_size:
            main_print("Resizing model embeddings to match tokenizer vocab size...")
            model.resize_token_embeddings(tokenizer_vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    main_print(f"Model size: {model_size/1e6:.2f}M parameters")

    # -----------------------
    # Data Collator
    # -----------------------
    def data_collator(features):
        # Some preprocessed data might not have 'labels' column
        # so we always create labels from input_ids, ignoring pad tokens
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        # Some preprocessed data might or might not have "attention_mask"
        if "attention_mask" in features[0]:
            attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        else:
            # If not present, create a mask with 1s where input_ids != pad_token_id
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

        labels = input_ids.clone()
        # Replace pad_token_id with -100 to ignore them in loss
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    # -----------------------
    # TrainingArguments
    # -----------------------
    output_path = "./out"

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.01,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        metric_for_best_model="loss" if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        bf16=True,
        bf16_full_eval=True,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=max(1, num_cpus-2),
        dataloader_pin_memory=True,
        torch_compile=True,
        gradient_checkpointing=False,
        optim='adamw_torch_fused',
        max_grad_norm=1.0,
        save_safetensors=False,
        log_level="info",
    )

    # -----------------------
    # Initialize Trainer
    # -----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # if you want validation metrics
    )

    # -----------------------
    # Train
    # -----------------------
    if args.resume_from_checkpoint:
        main_print(f"Resuming training from checkpoint {args.resume_from_checkpoint}...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.model_path:
        main_print(f"Continue training from pretrained model at {args.model_path}...")
        trainer.train()
    else:
        main_print("Starting training from scratch...")
        trainer.train()

    # -----------------------
    # Save final model & tokenizer
    # -----------------------
    main_print("Saving final model...")
    trainer.save_model(f"{output_path}/final_model")
    tokenizer.save_pretrained(f"{output_path}/final_model")

    if dist.is_initialized():
        dist.destroy_process_group()
