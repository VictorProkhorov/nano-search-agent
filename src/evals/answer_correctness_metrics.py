
import re
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
 
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

 




@dataclass
class CorrectnessMetrics:
    """Answer accuracy metrics"""
    exact_match: float
    token_f1: float
    semantic_similarity_sas: Optional[float] = None
    case_insensitive_em: float = 0.0
    substring_match: float = 0.0
    
    def to_dict(self):
        return {
            'exact_match': self.exact_match,
            'token_f1': self.token_f1,
            'semantic_similarity_sas': self.semantic_similarity_sas,
            'case_insensitive_em': self.case_insensitive_em,
        }
 


class CorrectnessEvaluator:
    """Compute answer correctness metrics"""
    def __init__(self, compute_semantic_sas: bool = False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        sas_model_name: str = "cross-encoder/stsb-roberta-large"
        self.sas_tokenizer = AutoTokenizer.from_pretrained(sas_model_name)
        self.sas_model = AutoModelForSequenceClassification.from_pretrained(sas_model_name).to(device)
        self.compute_semantic_sas = compute_semantic_sas

    
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
        
        inputs = self.sas_tokenizer(gold, pred, padding=True, return_tensors="pt")
        inputs = {k:v.to(self.sas_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.sas_model(**inputs)
        
        logit = outputs.logits[0].item()
        normalized_score = torch.sigmoid(torch.tensor(logit)).item()
        return float(normalized_score)
        
            
   
    def semantic_similarity_via_judge(self,
                                    gold: str,
                                    pred: Optional[str],
                                    ) -> float:
        """To Do"""
        pass
    
    def compute_correctness(self,
                           gold_answers: List[str], 
                           predicted_answers: List[str],
                           ) -> CorrectnessMetrics:
    
        """Compute all answer correctness metrics"""
        assert len(gold_answers) == len(predicted_answers), "Length mismatch"
        n = len(gold_answers)
        
        ems = [self.exact_match(g, p) for g, p in zip(gold_answers, predicted_answers)]
        f1s = [self.token_f1(g, p) for g, p in zip(gold_answers, predicted_answers)]
        ci_ems = [self.case_insensitive_match(g, p) for g, p in zip(gold_answers, predicted_answers)]
      
        
        sas_sims = None
        if self.compute_semantic_sas:
            sas_sims = [self.eval_semantic_similarity_sas(g, p) for g, p in zip(gold_answers, predicted_answers)]
            sas_sims = [s for s in sas_sims if s is not None]
            semantic_sim_sas = np.mean(sas_sims) if sas_sims else None
            
        else:
            semantic_sim_sas = None
        
        return CorrectnessMetrics(
            exact_match=np.mean(ems),
            token_f1=np.mean(f1s),
            semantic_similarity_sas=semantic_sim_sas,
            case_insensitive_em=np.mean(ci_ems),
        )
 
