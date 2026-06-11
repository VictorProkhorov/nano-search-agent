import re
import json

def load_trajectories(file_name:str):
    trajectories = []
    with open(file_name, 'r') as f:
        for line in f:
            trajectory = json.loads(line)
            trajectories.append(trajectory)
    return trajectories


def get_gold_answers(file_name:str):
    answers = []
    with open(file_name, 'r') as f:
        for answer in f:
            answers.append(answer)
    return answers


def get_pred_answers(trajectories:list):
    answers = []
    for idx, trajectory in enumerate(trajectories):
        content = ''
        for i in range(1, len(trajectory)):
            content += trajectory[i]['content']
        
        answer = extract_answer(content)
        answer = answer if answer == None else answer.strip()

        answers.append(answer)
    return answers


def extract_answer(s: str) -> str | None:
    match = re.search(r'<answer>([\s\S]*?)</answer>', s)
    print(match)
    return match.group(1).strip() if match else None


def eval(gold_answers:list,
        predicted_answers:list):
    score = 0
    for i, predicted_answer in enumerate(predicted_answers):
        gold_answer = gold_answers[i].strip().lower()
        predicted_answer = predicted_answer if predicted_answer == None else predicted_answer.lower()

        print('g:', gold_answer, 'p:', predicted_answer)
        if gold_answer == predicted_answer:
            score += 1    
    return score/len(predicted_answers)



def main():
    model_name = 'Qwen2.5-0.5B-Instruct-Search-LoRA'
    #model_name = 'Qwen2.5-0.5B-Instruct'
    dataset_split = 'validation'

    
    trajectories_file_name = f'../data/trajectories/hotpotqa_{model_name}_{dataset_split}.json'
    trajectories = load_trajectories(trajectories_file_name)
    
    gold_answers_file_name = f'../data/trajectories/hotpotqa_gold_answers_{model_name}_{dataset_split}.txt'
    gold_answers = get_gold_answers(gold_answers_file_name)

    pred_answers = get_pred_answers(trajectories)

    score = eval(gold_answers, pred_answers)
    print(f'EM: {score}')


if __name__ == '__main__':
    main()
