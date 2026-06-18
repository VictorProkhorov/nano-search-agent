import re
from functools import partial
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


PROMPT = """
Please answer the given question. You must conduct reasoning inside <think> and </think>
After reasoning, if you have enough knowledge to answer the question,
you can directly provide the answer inside <answer> and </answer> without
detailed illustrations. For example, <answer> xxx </answer>. Question: {question}"""


def get_model(model_id:str,
            device:torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def get_dataset(dataset_id:str):
    dataset = load_dataset(dataset_id, 'fullwiki')
    return dataset
    

def map_to_template(tokenizer, instance):
    question = instance['question']  # just a string
    question = PROMPT.format(question=question)  # no [0]
    question = [{"role": "user", "content": question}]
    question = tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
    instance['question'] = question
    return instance


def tokenize_map(tokenizer, instance):
    return tokenizer(instance['question'])


def preprocess_dataset(dataset:Dataset,
                      tokenizer):
    template_function = partial(map_to_template, tokenizer)
    new_dataset = dataset.map(template_function, batched=False)
    print(new_dataset)
    answers = dataset['validation']["answer"] 
    
    new_dataset = new_dataset.remove_columns(['id', 'type', 'level', 'supporting_facts', 'context', 'answer'])
    print(new_dataset)
    print(new_dataset['train'][0]['question'])


    tokenizer_function = partial(tokenize_map, tokenizer)
    tokenized_dataset = new_dataset.map(tokenizer_function, remove_columns=['question'])
    print(tokenized_dataset)

    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    test_dataset = DataLoader(tokenized_dataset['validation'], collate_fn = collate_fn, batch_size=1, shuffle=False)
    return test_dataset, answers



def extract_answers(decoded_texts):
    pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    return [
        m.group(1).strip() if (m := pattern.search(text)) else None
        for text in decoded_texts
    ]

@torch.no_grad()
def generae_answers(model:torch.Module,
        tokenizer,
        dataset:datase,
        device:torch.deivce):
    model.eval()
    predicted_answers = []
    for idx, batch in  enumerate(tqdm(dataset)):
        #print(batch)
        #exit()
        input_ids = batch['input_ids'].to(device)
        outputs = model.generate(input_ids, max_new_tokens=1024, num_beams=1, do_sample=False)
        #print(out)
        #exit()
        answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        #print(answer)
        final_answer = extract_answers([answer])
        #print(final_answer)
        predicted_answers.append(final_answer[0])
        if idx == 1000:
            break
    #print(predicted_answers)
    return predicted_answers

def eval(gold_answers, predicted_answers):
    score = 0
    for i, predicted_answer in enumerate(predicted_answers):
        gold_answer = gold_answers[i].lower()
        predicted_answer = predicted_answer if predicted_answer == None else predicted_answer.lower()

        print('g:', gold_answer, 'p:', predicted_answer)
        if gold_answer == predicted_answer:
            score += 1
    
    print(score/len(predicted_answers))



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    
    model, tokenizer = get_model(model_id, device)
    dataset_id = 'hotpotqa/hotpot_qa'
    dataset = get_dataset(dataset_id)

    test_dataset, gold_answers = preprocess_dataset(dataset, tokenizer)
    #print(gold_answers)
    #exit()
    predicted_answers = generae_answers(model, tokenizer, test_dataset, device)
    eval(gold_answers, predicted_answers)
    
    return


if __name__ == '__main__':
    main()