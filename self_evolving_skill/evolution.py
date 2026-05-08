import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
import hashlib


@dataclass
class EvolutionMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    total_interactions: int = 0
    successful_interactions: int = 0
    adaptation_rate: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class ParameterEvolution:
    def __init__(self, initial_params: Dict[str, float] = None):
        self.params = initial_params or {
            'relevance_weight': 0.4,
            'recency_weight': 0.3,
            'popularity_weight': 0.2,
            'diversity_weight': 0.1,
            'similarity_threshold': 0.7,
            'response_confidence_boost': 0.1
        }
        self.param_history: List[Dict] = []
        self.performance_log: List[Dict] = []
        
    def update_params(self, feedback: List[Tuple[str, float]], 
                     performance_delta: float):
        """
        根据反馈更新参数
        
        参数:
            feedback: [(参数名, 梯度方向)] 反馈元组列表
            performance_delta: 性能变化值
        """
        learning_rate = 0.1
        
        for param_name, gradient in feedback:
            if param_name in self.params:
                adjustment = gradient * learning_rate * performance_delta
                self.params[param_name] = np.clip(
                    self.params[param_name] + adjustment,
                    0.0, 1.0
                )
        
        self.param_history.append({
            'params': self.params.copy(),
            'performance_delta': performance_delta,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.param_history) > 1000:
            self.param_history = self.param_history[-1000:]

    def get_params(self) -> Dict[str, float]:
        return self.params.copy()
    
    def reset_to_defaults(self):
        self.params = {
            'relevance_weight': 0.4,
            'recency_weight': 0.3,
            'popularity_weight': 0.2,
            'diversity_weight': 0.1,
            'similarity_threshold': 0.7,
            'response_confidence_boost': 0.1
        }


class QueryUnderstandingModel:
    def __init__(self):
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        self.intent_classifiers: Dict[str, float] = {}
        self.entity_extractors: Dict[str, int] = defaultdict(int)
        self.context_window_size = 3
        
    def learn_from_interaction(self, query: str, success: bool, 
                              response_content: str):
        """
        从交互中学习查询模式
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        for i in range(len(words)):
            for window in range(1, min(self.context_window_size + 1, len(words) - i + 1)):
                pattern = ' '.join(words[i:i+window])
                self.query_patterns[pattern].append(
                    f"{'success' if success else 'failure'}:{response_content[:50]}"
                )
        
        for word in words:
            self.entity_extractors[word] += 1
        
        if success:
            self.intent_classifiers[query_lower] = \
                self.intent_classifiers.get(query_lower, 0) + 0.1
        else:
            self.intent_classifiers[query_lower] = \
                self.intent_classifiers.get(query_lower, 0) - 0.05
        
        self.intent_classifiers[query_lower] = np.clip(
            self.intent_classifiers.get(query_lower, 0), 0, 1
        )

    def predict_intent(self, query: str) -> str:
        """预测查询意图"""
        query_lower = query.lower()
        
        best_match = 'general'
        best_score = 0.0
        
        for pattern in self.query_patterns:
            if pattern in query_lower:
                success_count = sum(
                    1 for outcome in self.query_patterns[pattern]
                    if outcome.startswith('success')
                )
                total = len(self.query_patterns[pattern])
                if total > 0:
                    score = success_count / total
                    if score > best_score:
                        best_score = score
                        best_match = pattern
        
        return best_match
    
    def suggest_improvements(self) -> List[str]:
        """建议需要改进的地方"""
        suggestions = []
        
        low_performance_patterns = [
            pattern for pattern, outcomes in self.query_patterns.items()
            if len(outcomes) >= 3
            and sum(1 for o in outcomes if o.startswith('success')) / len(outcomes) < 0.3
        ]
        
        if low_performance_patterns:
            suggestions.append(f"注意这些低性能模式: {', '.join(low_performance_patterns[:5])}")
        
        return suggestions


class SelfEvolvingCore:
    def __init__(self, storage_path: str = "evolution_state.json"):
        self.metrics = EvolutionMetrics()
        self.param_evolution = ParameterEvolution()
        self.query_model = QueryUnderstandingModel()
        self.storage_path = storage_path
        self.interaction_buffer: List[Dict] = []
        self.evolution_threshold = 10
        self._load()
        
    def _load(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metrics = EvolutionMetrics(**data.get('metrics', {}))
                self.param_evolution = ParameterEvolution(
                    data.get('params', {}).get('params')
                )
                self.query_model.query_patterns = defaultdict(
                    list, data.get('query_patterns', {})
                )
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
    def _save(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': self.metrics.__dict__,
                'params': self.param_evolution.params,
                'query_patterns': dict(self.query_model.query_patterns)
            }, f, ensure_ascii=False, indent=2)
    
    def record_interaction(self, query: str, response: str, 
                          feedback_score: float, context: Dict = None):
        """
        记录交互并更新模型
        """
        success = feedback_score >= 0.5
        
        self.metrics.total_interactions += 1
        if success:
            self.metrics.successful_interactions += 1
        
        self.metrics.accuracy = (
            self.metrics.successful_interactions / self.metrics.total_interactions
        )
        
        self.query_model.learn_from_interaction(query, success, response)
        
        self.interaction_buffer.append({
            'query': query,
            'response': response,
            'feedback_score': feedback_score,
            'success': success,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.interaction_buffer) >= self.evolution_threshold:
            self._trigger_evolution()
    
    def _trigger_evolution(self):
        """
        触发自我进化过程
        """
        recent_interactions = self.interaction_buffer[-self.evolution_threshold:]
        
        successes = sum(1 for i in recent_interactions if i['success'])
        recent_accuracy = successes / len(recent_interactions)
        
        performance_delta = recent_accuracy - self.metrics.accuracy
        
        feedback_list = [
            ('relevance_weight', 0.1 if recent_accuracy > 0.7 else -0.1),
            ('recency_weight', 0.05 if recent_accuracy > 0.6 else -0.05),
            ('similarity_threshold', 0.02 if recent_accuracy > 0.5 else -0.02)
        ]
        
        self.param_evolution.update_params(feedback_list, performance_delta)
        
        self.metrics.adaptation_rate = abs(performance_delta)
        self.metrics.last_updated = datetime.now().isoformat()
        
        if len(recent_interactions) > 0:
            true_positives = sum(
                1 for i in recent_interactions 
                if i['success'] and i['feedback_score'] >= 0.7
            )
            predicted_positives = sum(
                1 for i in recent_interactions if i['feedback_score'] >= 0.5
            )
            
            if predicted_positives > 0:
                self.metrics.precision = true_positives / predicted_positives
            
            if successes > 0:
                self.metrics.recall = true_positives / successes
            
            if self.metrics.precision + self.metrics.recall > 0:
                self.metrics.f1_score = (
                    2 * self.metrics.precision * self.metrics.recall /
                    (self.metrics.precision + self.metrics.recall)
                )
        
        self._save()
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        获取优化建议
        """
        return {
            'current_params': self.param_evolution.get_params(),
            'metrics': {
                'accuracy': self.metrics.accuracy,
                'precision': self.metrics.precision,
                'recall': self.metrics.recall,
                'f1_score': self.metrics.f1_score,
                'total_interactions': self.metrics.total_interactions
            },
            'suggestions': self.query_model.suggest_improvements(),
            'adaptation_rate': self.metrics.adaptation_rate
        }
    
    def adapt_to_context(self, context_type: str) -> Dict[str, float]:
        """
        根据上下文类型调整参数
        """
        base_params = self.param_evolution.get_params()
        
        context_adjustments = {
            'technical': {
                'relevance_weight': 0.1,
                'similarity_threshold': 0.05
            },
            'creative': {
                'diversity_weight': 0.15,
                'relevance_weight': -0.1
            },
            'detailed': {
                'recency_weight': 0.1,
                'similarity_threshold': -0.05
            }
        }
        
        if context_type in context_adjustments:
            adjustments = context_adjustments[context_type]
            return {
                k: np.clip(v + adjustments.get(k, 0), 0, 1) 
                for k, v in base_params.items()
            }
        
        return base_params
