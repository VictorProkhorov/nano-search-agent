import re
import json
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset


def load_trajectories(file_name:str):
    trajectories = []
    with open(file_name, 'r') as f:
        for line in f:
            trajectory = json.loads(line)
            trajectories.append(trajectory)
    return trajectories


# Allowed tag names
ALLOWED_TAGS = {'think', 'tool_call', 'tool_response', 'answer'}
 
# Detects any XML-like opening tag
ANY_TAG_PATTERN = re.compile(r'<([a-zA-Z_][a-zA-Z0-9_]*)>')
 
# Extracts all blocks in order
BLOCK_PATTERN = re.compile(
    r'<(think|tool_call|tool_response|answer)>([\s\S]*?)</\1>'
)
 
# Full format: starts with <think>, ends with <answer>, nothing after </answer>
# Between the two, any sequence of known tagged blocks or plain text is fine.
# We validate structure by parsing tokens rather than a single regex,
# since allowing arbitrary plain text between tags makes a pure regex unwieldy.
 
 
@dataclass
class ValidationResult:
    is_valid: bool
    error: str | None
    blocks: list[tuple[str, str]]  # list of (tag_name, content)
 
 
def validate(s: str) -> ValidationResult:
    """
    Validate that a string matches the required format and return a detailed result.
    """
    s = s.strip()
    blocks = BLOCK_PATTERN.findall(s)
 
    # Guard: exactly one <answer> block allowed
    if s.count('<answer>') > 1:
        return ValidationResult(
            is_valid=False,
            error="Only one <answer> block is allowed, and it must be last.",
            blocks=blocks,
        )
 
    # Must end with </answer>
    if not s.endswith('</answer>'):
        return ValidationResult(
            is_valid=False,
            error="String must end with an <answer>...</answer> block.",
            blocks=blocks,
        )
 
    # Must start with <think>
    if not s.startswith('<think>'):
        return ValidationResult(
            is_valid=False,
            error="String must start with a <think>...</think> block.",
            blocks=blocks,
        )
 
    # No unknown tags allowed anywhere
    for tag in ANY_TAG_PATTERN.findall(s):
        if tag not in ALLOWED_TAGS:
            return ValidationResult(
                is_valid=False,
                error=f"Unknown tag <{tag}> found. Only think, tool_call, tool_response, and answer are allowed.",
                blocks=blocks,
            )
 
    # No content after </answer>
    answer_end = s.rfind('</answer>')
    if answer_end + len('</answer>') < len(s):
        return ValidationResult(
            is_valid=False,
            error="No content is allowed after the </answer> closing tag.",
            blocks=blocks,
        )
 
    # Validate all known tags are properly closed (no unclosed tags)
    for tag in ALLOWED_TAGS:
        opens = s.count(f'<{tag}>')
        closes = s.count(f'</{tag}>')
        if opens != closes:
            return ValidationResult(
                is_valid=False,
                error=f"Mismatched <{tag}> tags: {opens} opening vs {closes} closing.",
                blocks=blocks,
            )
 
    return ValidationResult(is_valid=True, error=None, blocks=blocks)

 
def check_is_valid(s: str) -> bool:
    return validate(s.strip()).is_valid

def extract_answer(s: str) -> str | None:
    match = re.search(r'<answer>([\s\S]*?)</answer>', s)
    return match.group(1).strip() if match else None

def filter_trajectories(trajectories):
    model_id = 'Qwen/Qwen2.5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    counter = 0
    answers = []
    messages = []
    for idx, trajectory in enumerate(trajectories):
        content = ''
        #print(trajectory[1:])
        for i in range(1, len(trajectory)):
            content += trajectory[i]['content']
        #print(content)
        is_valid = check_is_valid(content)
        if is_valid:
            counter += 1
            #print(content)
            #print(tokenizer.apply_chat_template(
            #        trajectory,
            #        add_generation_prompt=False,
            #        tokenize=False,
            #        add_special_tokens=False
            #    ))
            answer = extract_answer(content)
            answers.append((answer.strip(), idx))
            messages.append(trajectory)
    print(counter)
    return answers, messages


def get_gold_answers(file_name:str):
    answers = []
    with open(file_name, 'r') as f:
        for answer in f:
            answers.append(answer)
    return answers


def filter_correct_answers(gold_answers,
                        predicted_answers,
                        messages):
    score = 0
    messages_with_correct_answers = []
    for pred_answer_idx, predicted in enumerate(predicted_answers):
        predicted_answer, i = predicted
        gold_answer = gold_answers[i].strip().lower()
        predicted_answer = predicted_answer if predicted_answer == None else predicted_answer.lower()

        print('g:', gold_answer, 'p:', predicted_answer)
        if gold_answer == predicted_answer:
            messages_with_correct_answers.append(messages[pred_answer_idx])
            score += 1
    
    print(score/len(predicted_answers))
    return messages_with_correct_answers


def gen_dataset(messages):
    for message in messages:
        yield {'message':message}


def save_filtered_trajectories(messages:list,
                               path:str):
    dataset = Dataset.from_generator(gen_dataset, gen_kwargs={"messages": messages})
    print(dataset)
    #print(dataset['message'][1])
    dataset.save_to_disk(path)
    return

def main():
    model_name = 'Qwen2.5-7B-Instruct'
    dataset_split = 'train'
    
    file_name = f'../data/trajectories/hotpotqa_{model_name}_{dataset_split}.json'#'../data/trajectories/hotpotqa.json'
    
    trajectories = load_trajectories(file_name)
    print(len(trajectories))
    predicted_answers, messages = filter_trajectories(trajectories)

    file_name = f'../data/trajectories/hotpotqa_gold_answers_{model_name}_{dataset_split}.txt'#'../data/trajectories/hotpotqa_gold_answers.txt'
    gold_answers = get_gold_answers(file_name)
    messages = filter_correct_answers(gold_answers, predicted_answers, messages)
    print(len(messages))

    path_to_dataset = '../data/trajectories/filtered/hotpotqa_filtered_trajectories'
    save_filtered_trajectories(messages,
                               path_to_dataset)


if __name__ == '__main__':
    main()