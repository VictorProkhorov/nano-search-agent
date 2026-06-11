import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
from functools import partial

import torch
from torch.utils.data import DataLoader

import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk, Dataset

from tools import PartyPlanningRetrieverTool, RETRIVAL_SCHEMA, ddg_search, SEARCH_SCHEMA


def get_dataset(dataset_id:str):
    dataset = load_from_disk(dataset_id)
    return dataset


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ignore_start_tag = '<tool_response>'
        self.ignore_end_tag = '</tool_response>'

        self.ignore_pattern = re.compile(
            re.escape(self.ignore_start_tag) + r'.*?' + re.escape(self.ignore_end_tag),
            re.DOTALL
        )
    
    def __call__(self, batch):
        input_encodings, batch_chat_text = self.get_input_encodings(batch)
        labels = self.get_labels(input_encodings, batch_chat_text)
        return {'input_ids': input_encodings['input_ids'], 'attention_mask': input_encodings['attention_mask'], 'labels': labels}
    
    def get_input_encodings(self, batch):
        batch_chat_text = []
        for instance in batch:
            chat_text = self.tokenizer.apply_chat_template(
                    instance['message'],
                    tools=[SEARCH_SCHEMA],
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
    lora_alpha=16,           # ratio 1:1 — stable
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj"         # MLP
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    model, tokenizer = get_model(model_id, device)
    dataset_id = './data/trajectories/filtered/hotpotqa_filtered_trajectories'
    dataset = get_dataset(dataset_id)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    collate_fn = Collator(tokenizer)
    
    model_path = './trained_models/Qwen2.5-0.5B-Instruct-Search-LoRA'
    args = TrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        logging_strategy="epoch",
        gradient_accumulation_steps=1,
        num_train_epochs=15,
        #weight_decay=0.01,
        lr_scheduler_type="constant",
        learning_rate=1e-5,
        save_strategy="no",
        #fp16=True,
        remove_unused_columns=False,
        push_to_hub=False,
        optim='paged_adamw_8bit')
    
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