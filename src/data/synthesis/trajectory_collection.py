from pathlib import Path
import re, sys, asyncio
from tqdm import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema
from datasets import load_dataset, Dataset, DatasetDict

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
 
from ...mcp_tool_converter import fetch_tools_from_mcp, print_tool_schemas

"""
PROMPT = Answer the given question. You MUST conduct reasoning inside <think> and </think>
first. After reasoning, if you find you lack some
knowledge, you can call a search engine by <tool_call> query </tool_call>, and it will
return the top searched results between <tool_response> and </tool_response>. You
can search as many times as you want. If you find no further external knowledge
needed, provide the answer inside <answer> and </answer> without
detailed illustrations and the answer should be no more than three words. For example, <answer> xxx </answer>. Question: {question}
"""

PROMPT = """Answer the given question by following the provided instructions bellow.

INSTRUCTIONS:
1. First, conduct reasoning inside <think> and </think>
2. If you lack knowledge, call search engine with <tool_call>
3. Repeat thinking and searching until you have enough information
4. Synthesize results into a final answer

FORMAT:

<think>
[Your reasoning and analysis]
</think>

Then execute:
- Use <tool_call> query </tool_call> to search when needed
- Results appear between <tool_response> and </tool_response>
- Repeat <think> and <tool_call> as needed

Finally, provide your answer:
<answer>answer in no more than 3 words</answer>

Question: {question}"""


def get_model(model_id:str,
             device:torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer



def get_dataset(dataset_id:str):
    return load_dataset(dataset_id, 'fullwiki')


def get_question_sample(dataset:DatasetDict,
                        sample_size:int=1000, split:str='train')->Dataset:
    dataset = dataset.remove_columns(['id', 'type', 'level', 'supporting_facts', 'context'])[split]
    num_questions = len(dataset)
    g = torch.Generator()
    g.manual_seed(42)
    sample_ids = torch.randperm(num_questions, generator=g)[:sample_size]
    data_sample = dataset[sample_ids]
    return data_sample


def build_initial_messages(question: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": PROMPT.format(question=question)},
    ]


async def run_searches(query_list: List[str], tool_name: str, mcp_session: ClientSession) -> str:
    try:
        if tool_name == 'search':
            result = await mcp_session.call_tool("search", {"query_list": query_list})
            return result.content[0].text
        
        elif tool_name == 'retriever':
            if isinstance(query_list, list):
                query = query_list[0] if query_list else ""
            else:
                query = query_list
            
            result = await mcp_session.call_tool("retriever", {"query": query})
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
                # ── Tokenise using the native Qwen chat template with tool schema ──────
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    add_special_tokens=False
                )
                #print(f"\n===== prompt(turn {turn + 1}) =====\n")
                #print(prompt_text)
                #exit()
                enc = tokenizer(prompt_text, return_tensors="pt").to(device)
                print(enc['input_ids'].shape)
                if enc['input_ids'].shape[1] > 3100:
                    break

                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7,
                    do_sample=True)

                delta = tokenizer.decode(
                    out[0][enc.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                #print(f"\n===== Assistant (turn {turn + 1}) =====\n{delta}\n")

                # ── Append assistant turn to message history ──────────────────────────
                messages.append({"role": "assistant", "content": delta})

                # ── Check for a tool call ─────────────────────────────────────────────
                tool_name, tool_args = await extract_tool_call(delta)

                if tool_name != "search" and tool_name != "retriever":
                    #print('TOOL NAME', tool_name)
                    # No tool call → model is done
                    #print("No tool call detected — stopping.")
                    break
            
                # ── Execute the search ────────────────────────────────────────────────
                if tool_name == 'search':
                    query_list = tool_args.get("query_list", [])
                elif tool_name == "retriever":
                    query_list = tool_args.get("query", str)
                #print(f"  → Searching: {query_list}")
                results_text = await run_searches(query_list, tool_name, mcp_session)

                # ── Feed results back as a "tool" role message ────────────────────────
                # This is what Qwen2.5-Instruct's template expects (not a raw string)
                messages.append({
                    "role": "tool",
                    "name": "search",           # must match the function name in the schema
                    "content": results_text,
                })
            #print(messages)
            print("\n===== Final message history =====")
            for m in messages:
                role = m["role"].upper()
                content = (m["content"] or "")#[:200]  # truncate for display
                print(f"[{role}] {content}…\n")
            exit()
            f.write(json.dumps(messages) + "\n")
    return

def save_gold_answers(dataset, file_name:str):
    answers = dataset['answer']
    with open(file_name, 'w') as f:
        for answer in answers:
            f.write(answer + '\n')
    return None


async def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_id = 'Qwen/Qwen2.5-7B-Instruct'
    model_name = 'Qwen2.5-7B-Instruct'
    dataset_split = 'train'
    
    #model_id = f'{project_root}/models/Qwen2.5-0.5B-Instruct-Search-LoRA'
    #model_name = 'Qwen2.5-0.5B-Instruct-Search-LoRA'

    #dataset_split = 'validation'
    
    model, tokenizer = get_model(model_id, device)
    #print(tokenizer.chat_template)
    #exit()
    dataset_id = 'hotpotqa/hotpot_qa'
    dataset = get_dataset(dataset_id)
    dataset_sample = get_question_sample(dataset, sample_size=100, split=dataset_split)
    
    script_location = Path(__file__) 
    project_root = script_location.parent.parent.parent.parent 
    print(project_root)
    file_name = f'{project_root}/data/trajectories/hotpotqa_gold_answers_{model_name}_{dataset_split}.txt'
    save_gold_answers(dataset_sample, file_name)
    #print(tool.tool_schema)
    print("Connecting to MCP server...", file=sys.stderr)
    server_params = StdioServerParameters(
        command="python3",
        args=[f"{project_root}/src/tools.py"]
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()
            print("MCP server connected!", file=sys.stderr)

            
            print("\nFetching tools from MCP server...", file=sys.stderr)
            tools = await fetch_tools_from_mcp(mcp_session)
            
            #print(tools[1])
            #exit()
            print(f"Found {len(tools)} tools:", file=sys.stderr)
            for tool in tools:
                print(f"  - {tool['function']['name']}: {tool['function']['description']}", file=sys.stderr)
            

            file_name = f'{project_root}/data/trajectories/hotpotqa_{model_name}_{dataset_split}.json'
            await unroll(model,
                    tokenizer,
                    dataset_sample,
                    device,
                    mcp_session,
                    tools=[tools[1]],
                    file_name=file_name)
    
    print("Done!", file=sys.stderr)
    return
    
  

if __name__ == '__main__':
    asyncio.run(main())