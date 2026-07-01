
import re
import json
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
 
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer

 

JUDGE_PROMPT = """You are an expert evaluator comparing a predicted answer to a gold reference answer.
 
Your task: Rate how well the predicted answer matches the gold answer on a scale of 1 to 4.
 
Evaluation criteria:
- Factual alignment: Does the predicted answer contain the same key facts as the gold answer?
- Coverage: Does it cover the main points in the gold answer?
- Accuracy: Are there any factual errors or contradictions with the gold answer?
- Completeness: Does it capture the core content, or are significant aspects missing?
 
Scale:
1: Factually incorrect, contradicts gold answer, or completely misses the core content
2: Contains some correct elements but misses key facts or has significant gaps compared to gold
3: Mostly aligns with gold answer but lacks detail or has minor omissions
4: Accurately captures the gold answer—same key facts, well-articulated, complete
 
Question: {question}
 
Gold Answer: {gold_answer}
 
Predicted Answer: {predicted_answer}
 
Compare the predicted answer to the gold answer. Provide your evaluation in exactly this format:
Evaluation: [How well does predicted match gold? What's correct/missing/wrong?]
Rating: [1, 2, 3, or 4]"""


ATTRIBUTION_PROPMT = """Given the following retrieved documents and a question, determine if the answer is supported/entailed by the content given the question.
 
Question: {question}

Retrieved Documents: {retrieved_content}
 
Answer: {answer}
 
Answer only with "Yes" or "No". The answer is supported if it can be inferred from the retrieved documents given the question.
 
Answer:"""


@dataclass
class Answer_Metrics:
    """Answer accuracy metrics"""
    exact_match: float
    token_f1: float
    semantic_similarity_sas: Optional[float] = None
    case_insensitive_em: float = 0.0
    semantic_similarity_judge: Optional[float] = None
    attribution_score: Optional[float] = None 
    
    def to_dict(self):
        return {
            'exact_match': self.exact_match,
            'token_f1': self.token_f1,
            'semantic_similarity_sas': self.semantic_similarity_sas,
            'semantic_similarity_judge': self.semantic_similarity_judge,
            'case_insensitive_em': self.case_insensitive_em,
            'attribution_score': self.attribution_score
        }
 


class Answer_Evaluator:
    """Compute answer correctness metrics"""
    def __init__(self,
                compute_semantic_sas: bool = False,
                compute_semantic_judge: bool = False,
                compute_attribution:bool = False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # SAS Model
        sas_model_name = "cross-encoder/stsb-roberta-large"
        self.sas_tokenizer,  self.sas_model = self._load_model(sas_model_name)
        self.compute_semantic_sas = compute_semantic_sas

        # Judge Model
        judge_model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.judge_tokenizer,  self.judge_model = self._load_model(judge_model_name, is_causal=True)
        self.compute_semantic_judge = compute_semantic_judge

        self.compute_attribution = compute_attribution

    def _load_model(self,
                    model_name:str,
                    is_causal: bool = False):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if is_causal:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        model.eval()
        return tokenizer, model

    
    def exact_match(self,
                    gold: str,
                    pred: str) -> float:
        """Exact match after normalization"""
        if pred is None:
            return 0.0
        return float(gold.strip() == pred.strip())
    
    
    def case_insensitive_match(self,
                            gold:str,
                            pred:str) -> float:
        """Case-insensitive exact match"""
        if pred is None:
            return 0.0
        return float(gold.strip().lower() == pred.strip().lower())
 
    
    def token_f1(self,
                gold: str,
                pred: str) -> float:
        """Token-level F1 score"""
        if pred is None:
            return 0.0
        
        gold_tokens = set(gold.lower().split())
        pred_tokens = set(pred.lower().split())
        
        if len(gold_tokens) == 0 or len(pred_tokens) == 0:
            return 0.0
        
        common = gold_tokens & pred_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    

    def eval_semantic_similarity_sas(self,
                                    gold: str,
                                    pred: Optional[str]
                                    ) -> float:
        if pred is None:
            return None
        
        inputs = self.sas_tokenizer(gold, pred, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sas_model(**inputs)
        
        logit = outputs.logits[0].item()
        normalized_score = torch.sigmoid(torch.tensor(logit)).item()
        return float(normalized_score)
        
            
   
    def eval_semantic_similarity_judge(self,
                                    question:str,
                                    gold: str,
                                    pred: str,
                                    ) -> float:
        prompt = JUDGE_PROMPT.format(
                question=question,
                gold_answer=gold,
                predicted_answer=pred,
            )
            
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.judge_model.generate(
                                **inputs,
                                max_new_tokens=150,
                                do_sample=False)
            
        judge_response = self.judge_tokenizer.decode(
                            outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
            
        rating = self._extract_rating(judge_response)
        #print(rating)
        return rating/4

    def _extract_rating(self, response: str) -> Optional[float]:
   
        """
        Extract numeric rating from judge response.
        
        Looks for 'Rating: N' where N is 1-4.
        """
        try:
            for line in response.split("\n"):
                if "Rating:" in line or "rating:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        rating_str = parts[-1].strip()
                        rating = float(rating_str.split()[0])
                        if 1.0 <= rating <= 4.0:
                            return rating
            
            return 0.
        
        except Exception:
            return 0.

    def eval_answer_attribution(self,
                                question:str,
                                traj:List[dict],
                                pred_answer:str) -> int:
        retrieved_content = self._extract_content(traj)
        if retrieved_content == '':
            return 0
        
        prompt = ATTRIBUTION_PROPMT.format(
                question=question,
                retrieved_content=retrieved_content,
                answer=pred_answer,
            )
            
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.judge_model.generate(
                                **inputs,
                                max_new_tokens=15,
                                do_sample=False)
            
        response = self.judge_tokenizer.decode(
                            outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        # Parse response
        if response.upper().startswith('YES'):
            return 1
        elif response.upper().startswith('NO'):
            return 0
        else:
            return 0

    
    def _extract_content(self, trajectory: List[Dict]) -> str:
        """Extract all retrieved/searched content from trajectory"""
        content = []
        for msg in trajectory:
            if msg.get('role') == 'tool':
                content.append(msg.get('content', ''))
        return '\n'.join(content)

    
    def compute(self,
                gold_answers: List[str], 
                predicted_answers: List[str],
                questions: List[str],
                trajectories: List[List[dict]]
                ) -> CorrectnessMetrics:
    
        """Compute all answer correctness metrics"""
        assert len(gold_answers) == len(predicted_answers), "Length mismatch"
        n = len(gold_answers)
        
        ems = [self.exact_match(g, p) for g, p in zip(gold_answers, predicted_answers)]
        f1s = [self.token_f1(g, p) for g, p in zip(gold_answers, predicted_answers)]
        ci_ems = [self.case_insensitive_match(g, p) for g, p in zip(gold_answers, predicted_answers)]
      
        
        if self.compute_semantic_sas:
            sas_sims = [self.eval_semantic_similarity_sas(g, p) for g, p in zip(gold_answers, predicted_answers)]
            sas_sims = [s for s in sas_sims if s is not None]
            semantic_sim_sas = np.mean(sas_sims) if sas_sims else None
            
        else:
            semantic_sim_sas = None

        if self.compute_semantic_judge:
            judge_sims = [self.eval_semantic_similarity_judge(q, g, p) for g, p, q in tqdm(zip(gold_answers, predicted_answers, questions))]
            #judge_sims = [s for s in judge_sims if s is not None]
            semantic_sim_judge = np.mean(judge_sims) if judge_sims else None
        else:
            semantic_sim_judge = None

        if self.compute_attribution:
            attribution_scores = [self.eval_answer_attribution(question, traj, pred_answer)\
                                                        for question, traj, pred_answer in tqdm(zip(questions, trajectories, predicted_answers))]
            attribution_score = np.mean(attribution_scores) if attribution_scores else None
        else:
            attribution_score = None

        return Answer_Metrics(
            exact_match=np.mean(ems),
            token_f1=np.mean(f1s),
            semantic_similarity_sas=semantic_sim_sas,
            semantic_similarity_judge=semantic_sim_judge,
            case_insensitive_em=np.mean(ci_ems),
            attribution_score = attribution_score
        )
 
