import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field

from .answer_metrics import Answer_Evaluator, Answer_Metrics
from .trajectory_metrics import Trajectory_Evaluator, Trajectory_Metrics


@dataclass
class Eval_Result:
    """Full evaluation result"""
    answer: Answer_Metrics
    trajectory: Trajectory_Metrics
    
    def to_dict(self) -> Dict:
        return {
            'answer': self.answer.to_dict(),
            'trajectory': self.trajectory.to_dict(),
        }
        

    def summary(self) -> str:
        """Human-readable summary"""
        return f"""
                ╔════════════════════════════════════════════╗
                ║             EVALUATION SUMMARY             ║
                ╚════════════════════════════════════════════╝
 
                ANSWER
                    Token F1:           {self.answer.token_f1:.3f}
                    Exact Match (EM):        {self.answer.exact_match:.3f}
                    Case-Insensitive (EM):   {self.answer.case_insensitive_em:.3f}
                    Semantic Sim (Semantic Answer Similarity):       {self.answer.semantic_similarity_sas or 'N/A'}
                    Semantic Sim (LLM-as-a-Judge):       {self.answer.semantic_similarity_judge or 'N/A'}
                    Attribution:                         {self.answer.attribution_score or 'N/A'}
                    
                TRAJECTORY
                    Answer Extraction Rate:             {self.trajectory.answer_format_rate:.3f}
                    % of Trajectories with Reasoning:   {self.trajectory.has_reasoning:.3f}
                    Av. Num. Reasoning Steps:           {self.trajectory.av_num_reasoning_steps: .3f}
                    % of Trajectories with a Tool Call: {self.trajectory.has_tool_call: .3f}
                    Av. Num. Tool Calls:                {self.trajectory.av_num_tool_calls: .3f}
                    Tool Error Rate:                    {self.trajectory.tool_error_rate: .3f}
                    Av. Num Convers. Turns:             {self.trajectory.av_num_turns_per_traj: .3f}     
                """

    


class Evaluator:
    """Main evaluation orchestrator"""
    def __init__(self,
                trajectories_path:str='',
                answers_path:str='',
                compute_semantic_sas: bool = False,
                compute_semantic_judge: bool = False,
                compute_attribution:bool = False):
    
        print("Loading trajectories...", file=sys.stderr)
        self.trajectories:  List[List[Dict]] = self.load_trajectories(trajectories_path)
    
        print("Loading gold answers...", file=sys.stderr)
        self.gold_answers: List[str] = self.load_gold_answers(answers_path)

        self.answer_correctness_evaluator = Answer_Evaluator(compute_semantic_sas=compute_semantic_sas,
                                                            compute_semantic_judge=compute_semantic_judge,
                                                            compute_attribution=compute_attribution)

        self.trajectory_evaluator = Trajectory_Evaluator()
    
    def evaluate(self) -> ComprehensiveEvalResult:
       
        assert len(self.trajectories) == len(self.gold_answers), "Length mismatch"
        
        # Extract predicted answers
        predicted_answers = [
            self.extract_answer(traj) 
            for traj in self.trajectories
        ]
        # Extract questions
        questions = [
            self.extract_question(traj)
            for traj in self.trajectories
        ]
        
        # Compute metrics
        answer_correctness = self.answer_correctness_evaluator.compute(
            self.gold_answers, predicted_answers, questions, self.trajectories
        
        )
        trajectory = self.trajectory_evaluator.compute(self.trajectories)

        return Eval_Result(
            answer=answer_correctness,
            trajectory=trajectory,
        )


    def load_trajectories(self, file_path: str) -> List[List[Dict]]:
        """Load trajectory data (JSONL format)"""
        trajectories = []
        with open(file_path, 'r') as f:
            for line in f:
                trajectory = json.loads(line)
                trajectories.append(trajectory)
        return trajectories
 

    def load_gold_answers(self, file_path: str) -> List[str]:
        """Load gold answer data (one per line)"""
        answers = []
        with open(file_path, 'r') as f:
            for line in f:
                answers.append(line.strip())
        return answers
 
    
    def extract_answer(self, trajectory: List[Dict]) -> Optional[str]:
        """Extract final answer from trajectory"""
        for msg in reversed(trajectory):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                match = re.search(r'<answer>([\s\S]*?)</answer>', content)
                if match:
                    return match.group(1).strip()
        return None
    
    def extract_question(self, trajectory: List[Dict]):
        question = trajectory[0]['content'].split('Question:')[1].strip()
        return question
    
    
 

def main():
    model_name = 'Qwen2.5-0.5B-Instruct-Search-LoRA'
    #model_name = 'Qwen2.5-0.5B-Instruct'
    dataset_split = 'validation'

    base_dir = Path(__file__).parent.parent.parent
    trajectories_path = f'{base_dir}/data/trajectories/hotpotqa_{model_name}_{dataset_split}.json'
    gold_answers_path = f'{base_dir}/data/trajectories/hotpotqa_gold_answers_{model_name}_{dataset_split}.txt'
    
    evaluator = Evaluator(trajectories_path=trajectories_path,
                        answers_path=gold_answers_path,
                        compute_semantic_sas=True,
                        compute_semantic_judge=False,
                        compute_attribution=True)
    result = evaluator.evaluate()
    print(result.summary())


if __name__ == '__main__':
    main()