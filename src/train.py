import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

#import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk, Dataset

import argparse
from hydra import initialize, compose
from omegaconf import OmegaConf

from .mcp_tool_converter import fetch_tools_from_mcp


import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


def get_dataset(dataset_id:str):
    dataset = load_from_disk(dataset_id)
    return dataset


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="config")
    parser.add_argument("overrides", nargs="*", default=[],
                       help="Override config values: key=value")
    args = parser.parse_args()
    with initialize(version_base=None, config_path="./conf"):
        config = compose(config_name=args.config_name, 
                     overrides=args.overrides)  # specify your default config
        print(config)
    return config


def get_tools_sync(path):
    """Synchronous wrapper to fetch tools from MCP server"""
    async def fetch():
        server_params = StdioServerParameters(
            command="python3",
            args=[f"{path}/src/tools.py"]
        )
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await fetch_tools_from_mcp(session)
                return tools
    
    return asyncio.run(fetch())


class Collator:
    def __init__(self, tokenizer, tools):
        self.tokenizer = tokenizer
        self.ignore_start_tag = '<tool_response>'
        self.ignore_end_tag = '</tool_response>'

        self.ignore_pattern = re.compile(
            re.escape(self.ignore_start_tag) + r'.*?' + re.escape(self.ignore_end_tag),
            re.DOTALL
        )
        self.tools = tools
    
    def __call__(self, batch):
        input_encodings, batch_chat_text = self.get_input_encodings(batch)
        labels = self.get_labels(input_encodings, batch_chat_text)
        return {'input_ids': input_encodings['input_ids'], 'attention_mask': input_encodings['attention_mask'], 'labels': labels}
    
    def get_input_encodings(self, batch):
        batch_chat_text = []
        for instance in batch:
            chat_text = self.tokenizer.apply_chat_template(
                    instance['message'],
                    tools=self.tools,
                    add_generation_prompt=False,
                    tokenize=False,
                    add_special_tokens=False
                )
            batch_chat_text.append(chat_text)
        
        batch_encodings = self.tokenizer(
            batch_chat_text, padding=True,
            truncation=True, return_tensors="pt"
        )

        return batch_encodings, batch_chat_text

    
    def get_labels(self, input_encodings, batch_chat_text):
        ignore_idx = -100
        labels = input_encodings['input_ids'].clone()
        labels[input_encodings['attention_mask'] == 0] = ignore_idx

        batch_size = labels.shape[0]
        for i in range(batch_size):
            chat_text = batch_chat_text[i]
            prompt_str, response_str = chat_text.split("<|im_start|>assistant", 1)
            response_str = "<|im_start|>assistant" + response_str
            prompt_token_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
            labels[i, :prompt_token_len] = ignore_idx
            
            for match in self.ignore_pattern.finditer(response_str):   
                prefix_in_response = response_str[:match.start()]
                matched_block = match.group(0)
                    
                prefix_token_len = len(self.tokenizer.encode(prefix_in_response, add_special_tokens=False))
                block_token_len = len(self.tokenizer.encode(matched_block, add_special_tokens=False))

                mask_start_token_idx = prompt_token_len + prefix_token_len
                mask_end_token_idx = mask_start_token_idx + block_token_len
                labels[i, mask_start_token_idx:mask_end_token_idx] = ignore_idx

        return labels




def get_model(model_id:str,
            device:torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
   

    lora_config = LoraConfig(
    r=16,
    lora_alpha=16, # ratio 1:1 — stable
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # attention
        "gate_proj", "up_proj", "down_proj" # MLP
    ],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM
    )

    # 3. Create trainable PEFT model
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable() 
    model.print_trainable_parameters()
    return model, tokenizer



def main():
    script_dir = Path().parent.parent
    config = load_config()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_id = config.model.id
    model, tokenizer = get_model(model_id, device)
    
    dataset_id = f'{script_dir}{config.dataset.id}'
    dataset = get_dataset(dataset_id)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tools = get_tools_sync(script_dir)
    #tools = [tools[1]]
    collate_fn = Collator(tokenizer, tools)
    
    model_path = f'{script_dir}/models/Qwen2.5-0.5B-Instruct-Search-LoRA'
    args = TrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
        eval_strategy=config.trainer.eval_strategy,
        logging_strategy=config.trainer.logging_strategy,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        num_train_epochs=config.trainer.num_train_epochs,
        lr_scheduler_type=config.trainer.lr_scheduler_type,
        learning_rate=config.trainer.learning_rate,
        save_strategy=config.trainer.save_strategy,
        remove_unused_columns=config.trainer.remove_unused_columns,
        push_to_hub=config.trainer.push_to_hub,
        optim=config.trainer.optim)
    
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    #trainer.save_model()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return


if __name__ == '__main__':
    main()