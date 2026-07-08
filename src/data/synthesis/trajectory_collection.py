
# Standard library
import re
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List
 
# Third party
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from hydra import initialize, compose
from omegaconf import OmegaConf
from tqdm import tqdm

 
# Local application
from ...mcp_tool_converter import fetch_tools_from_mcp, print_tool_schemas
from .prompts import SEARCH_AGENT_PROMPT


# ============ MISC =========== #
def print_message_history(messages:List[dict]):

    print("\n===== Final message history =====")
    for m in messages:
        role = m["role"].upper()
        content = (m["content"] or "")
        print(f"[{role}] {content}\n")
    return None

    
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

# ============ DATA LOADING/HANDLING =========== #

def get_dataset(dataset_id:str):

    return load_dataset(dataset_id, 'fullwiki')


def get_question_sample(dataset:DatasetDict,
                        sample_size:int=1000,
                        split:str='train')->Dataset:

    dataset = dataset.remove_columns(['id', 'type', 'level', 'supporting_facts', 'context'])[split]
    num_questions = len(dataset)
    g = torch.Generator()
    g.manual_seed(42)
    sample_ids = torch.randperm(num_questions, generator=g)[:sample_size]
    data_sample = dataset[sample_ids]
    return data_sample


def save_gold_answers(dataset, file_name:str):

    answers = dataset['answer']
    with open(file_name, 'w') as f:
        for answer in answers:
            f.write(answer + '\n')
    return None


# ============ MODEL =========== #

def get_model(model_id:str,
             device:torch.device):

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def build_initial_messages(question: str) -> List[Dict[str, Any]]:

    return [
        {"role": "user", "content": SEARCH_AGENT_PROMPT.format(question=question)},
    ]


# ============ TOOL HANDLING =========== #

async def run_searches(query_list: List[str],
                      tool_name: str,
                      mcp_session: ClientSession) -> str:

    try:
        if tool_name == 'search':
            result = await mcp_session.call_tool("search", {"query_list": query_list})
            return result.content[0].text
    except Exception as e:
        return f"Error calling tool {tool_name}: {str(e)}"



async def extract_tool_call(decoded: str):

    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", decoded, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(1))
        return obj.get("name"), obj.get("arguments", {})
    except json.JSONDecodeError:
        return None, None


# ============ AGENT LOOP =========== #

async def unroll(model:torch.nn.Module,
                tokenizer,
                dataset:Dataset,
                device:torch.device,
                mcp_session,
                tools=[],
                file_name:str= ''):
    
    MAX_TURNS = 6
    MAX_NEW_TOKENS  = 512
    questions = dataset['question']
    with open(file_name, 'w') as f:

        for question_id, question in enumerate(tqdm(questions)):
            messages  = build_initial_messages(question)
            for turn in range(MAX_TURNS):
                # 1. Tokenise using the native Qwen chat template
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    add_special_tokens=False
                )
                
                encoded_prompt = tokenizer(prompt_text, return_tensors="pt").to(device)
                # 2. Generate a response                
                if encoded_prompt['input_ids'].shape[1] > 3100: # need to threshold otherwise OOM
                    break

                model_response = model.generate(
                    **encoded_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7,
                    do_sample=True)

                new_text = tokenizer.decode(
                    model_response[0][encoded_prompt.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            
                messages.append({"role": "assistant", "content": new_text})

                # 3. Check for a tool call
                tool_name, tool_args = await extract_tool_call(new_text)

                if tool_name != "search":
                    # No tool call -> model is done
                    break
            
                # 4. Execute the search
                if tool_name == 'search':
                    query_list = tool_args.get("query_list", [])
                
                results_text = await run_searches(query_list, tool_name, mcp_session)

                # 5. Feed results back as a "tool" role message
                messages.append({
                    "role": "tool",
                    "name": "search",           # must match the function name in the schema
                    "content": results_text,
                })
            # uncomment for debugging
            #print_message_history(messages)
            f.write(json.dumps(messages) + "\n")
    return None




async def main():
    # Misc
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = load_config()
    script_location = Path() 
    project_root = script_location.parent.parent.parent.parent 
    
    # Model
    model_id = str(project_root) + config.model.id if config.model.local_path else config.model.id
    model, tokenizer = get_model(model_id, device)
   
    # Data
    dataset = get_dataset(config.dataset.id)
    dataset_sample = get_question_sample(dataset,
                                        sample_size=config.traj.size,
                                        split=config.dataset.split)
    
    file_name = f'{project_root}/data/trajectories/hotpotqa_gold_answers_{config.model.name}_{config.dataset.split}.txt'
    save_gold_answers(dataset_sample, file_name)
    
    # MCP + Agent Loop
    print("Connecting to MCP server...")
    server_params = StdioServerParameters(
        command="python3",
        args=[f"{project_root}/src/tools.py"]
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()
            print("MCP server connected!")

            
            print("\nFetching tools from MCP server...")
            tools = await fetch_tools_from_mcp(mcp_session)
            
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['function']['name']}: {tool['function']['description']}")
            

            file_name = f'{project_root}/data/trajectories/hotpotqa_{config.model.name}_{config.dataset.split}.json'
            await unroll(model,
                    tokenizer,
                    dataset_sample,
                    device,
                    mcp_session,
                    tools=tools,
                    file_name=file_name)
    
    print("Done!")
    return
    

if __name__ == '__main__':
    asyncio.run(main())