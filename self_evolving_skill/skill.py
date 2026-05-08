import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter

from knowledge_base import KnowledgeBase, KnowledgeEntry, FeedbackType
from evolution import SelfEvolvingCore, EvolutionMetrics


@dataclass
class QueryContext:
    domain: str = "general"
    complexity: str = "medium"
    urgency: str = "normal"
    user_preference: Dict[str, float] = None

    def __post_init__(self):
        if self.user_preference is None:
            self.user_preference = {}


@dataclass
class QueryResult:
    entry: KnowledgeEntry
    score: float
    confidence: float
    reasoning: str


class SelfEvolvingSkill:
    def __init__(self, 
                 knowledge_base_path: str = "knowledge_base.json",
                 evolution_path: str = "evolution_state.json"):
        self.knowledge_base = KnowledgeBase(knowledge_base_path)
        self.evolution_core = SelfEvolvingCore(evolution_path)
        
        self.stopwords = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        ])
        
        self._initialize_default_knowledge()

    def _initialize_default_knowledge(self):
        """初始化默认知识库（如果为空）"""
        if len(self.knowledge_base.entries) == 0:
            default_entries = [
                {
                    'content': 'Self-evolving systems can improve their performance over time through machine learning.',
                    'tags': ['ml', 'self-improvement', 'ai'],
                    'source': 'system'
                },
                {
                    'content': 'Feedback loops are essential for self-improvement in intelligent systems.',
                    'tags': ['feedback', 'optimization', 'systems'],
                    'source': 'system'
                }
            ]
            
            for entry_data in default_entries:
                self.knowledge_base.add(**entry_data)

    def _tokenize(self, text: str) -> List[str]:
        """分词并去除停用词"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]

    def _calculate_similarity(self, query_tokens: List[str], 
                            entry_tokens: List[str]) -> float:
        """计算查询与条目的相似度"""
        if not query_tokens or not entry_tokens:
            return 0.0
        
        query_set = set(query_tokens)
        entry_set = set(entry_tokens)
        
        intersection = len(query_set & entry_set)
        union = len(query_set | entry_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _rank_results(self, query: str, 
                     context: QueryContext = None) -> List[QueryResult]:
        """根据多种因素对结果进行排名"""
        query_tokens = self._tokenize(query)
        params = self.evolution_core.param_evolution.get_params()
        
        if context:
            params = self.evolution_core.adapt_to_context(context.domain)
        
        results: List[QueryResult] = []
        
        for entry_id, entry in self.knowledge_base.entries.items():
            entry_tokens = self._tokenize(entry.content)
            
            relevance_score = self._calculate_similarity(query_tokens, entry_tokens)
            
            recency_score = 0.5
            if entry.last_accessed:
                days_since = (
                    datetime.now() - datetime.fromisoformat(entry.last_accessed)
                ).days
                recency_score = np.exp(-days_since / 30)
            
            popularity_score = entry.relevance_score
            
            diversity_score = 1.0
            if results:
                existing_tokens = []
                for r in results:
                    existing_tokens.extend(self._tokenize(r.entry.content))
                existing_set = set(existing_tokens)
                current_set = set(entry_tokens)
                overlap = len(existing_set & current_set) / max(len(existing_set | current_set), 1)
                diversity_score = 1 - overlap
            
            final_score = (
                params['relevance_weight'] * relevance_score +
                params['recency_weight'] * recency_score +
                params['popularity_weight'] * popularity_score +
                params['diversity_weight'] * diversity_score
            )
            
            confidence = min(final_score + params['response_confidence_boost'], 1.0)
            
            reasoning_parts = []
            if relevance_score > 0.3:
                reasoning_parts.append(f"内容相关性: {relevance_score:.2f}")
            if recency_score > 0.5:
                reasoning_parts.append(f"时效性: {recency_score:.2f}")
            if popularity_score > 0.6:
                reasoning_parts.append(f"历史表现: {popularity_score:.2f}")
            
            results.append(QueryResult(
                entry=entry,
                score=final_score,
                confidence=confidence,
                reasoning=", ".join(reasoning_parts) if reasoning_parts else "基础匹配"
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def query(self, question: str, 
             context: QueryContext = None,
             top_k: int = 3) -> Dict[str, Any]:
        """
        执行查询
        
        参数:
            question: 用户问题
            context: 查询上下文
            top_k: 返回结果数量
        
        返回:
            包含答案、置信度和元数据的字典
        """
        ranked_results = self._rank_results(question, context)
        
        if not ranked_results:
            return {
                'answer': 'I don\'t have enough information to answer this question.',
                'confidence': 0.0,
                'sources': [],
                'requires_human_input': True
            }
        
        top_results = ranked_results[:top_k]
        best_result = top_results[0]
        
        if best_result.confidence < 0.3:
            return {
                'answer': self._generate_fallback_response(question),
                'confidence': best_result.confidence,
                'sources': [r.entry.id for r in top_results],
                'requires_human_input': True
            }
        
        response_data = {
            'answer': best_result.entry.content,
            'confidence': best_result.confidence,
            'sources': [r.entry.id for r in top_results],
            'reasoning': best_result.reasoning,
            'related_entries': [
                {
                    'id': r.entry.id,
                    'content': r.entry.content[:100] + '...' if len(r.entry.content) > 100 else r.entry.content,
                    'score': r.score
                }
                for r in top_results[1:]
            ],
            'requires_human_input': False
        }
        
        self.evolution_core.record_interaction(
            query=question,
            response=best_result.entry.content,
            feedback_score=best_result.confidence,
            context={'domain': context.domain if context else 'general'}
        )
        
        return response_data

    def _generate_fallback_response(self, question: str) -> str:
        """生成后备响应"""
        partial_matches = self._rank_results(question)
        
        if partial_matches:
            similar = partial_matches[0].entry.content
            return f"I found some partial information: {similar[:200]}... Could you provide more details?"
        
        return "I need more information to answer this accurately. Could you rephrase your question?"

    def add_knowledge(self, content: str, 
                     tags: List[str] = None,
                     source: str = "manual") -> str:
        """添加新知识"""
        entry = self.knowledge_base.add(content, tags=tags, source=source)
        return entry.id

    def provide_feedback(self, query: str, 
                        entry_id: str,
                        feedback: FeedbackType,
                        correction: str = None):
        """提供反馈"""
        self.knowledge_base.update_feedback(entry_id, feedback, query)
        
        if feedback == FeedbackType.CORRECTION and correction:
            self.add_knowledge(
                content=correction,
                tags=['correction', 'learned'],
                source=f'correction_of_{entry_id}'
            )

    def learn_from_interaction(self, query: str, 
                              expected_response: str,
                              actual_response: str,
                              success: bool):
        """从交互中学习"""
        feedback_score = 1.0 if success else 0.0
        
        self.evolution_core.record_interaction(
            query=query,
            response=actual_response,
            feedback_score=feedback_score,
            context={'expected': expected_response}
        )
        
        if not success and expected_response:
            self.add_knowledge(
                content=expected_response,
                tags=['learned', 'improvement'],
                source='interaction_feedback'
            )

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        kb_stats = self.knowledge_base.get_stats()
        evolution_recommendations = self.evolution_core.get_recommendations()
        
        return {
            'knowledge_base': kb_stats,
            'evolution': evolution_recommendations,
            'system_ready': True,
            'total_entries': kb_stats['total_entries'],
            'performance_metrics': evolution_recommendations['metrics']
        }

    def auto_optimize(self) -> Dict[str, Any]:
        """自动优化"""
        pruned_count = self.knowledge_base.prune_low_value(threshold=0.2)
        
        recommendations = self.evolution_core.get_recommendations()
        
        return {
            'pruned_entries': pruned_count,
            'current_metrics': recommendations['metrics'],
            'suggestions': recommendations['suggestions'],
            'optimization_complete': True
        }

    def export_learning(self) -> Dict[str, Any]:
        """导出学习成果"""
        return {
            'knowledge_entries': [
                {
                    'id': e.id,
                    'content': e.content,
                    'tags': e.tags,
                    'relevance_score': e.relevance_score
                }
                for e in self.knowledge_base.entries.values()
            ],
            'learned_parameters': self.evolution_core.param_evolution.get_params(),
            'metrics': self.evolution_core.metrics.__dict__
        }

    def import_knowledge(self, knowledge_data: List[Dict]):
        """导入知识"""
        for item in knowledge_data:
            if 'content' in item:
                self.add_knowledge(
                    content=item['content'],
                    tags=item.get('tags', []),
                    source=item.get('source', 'imported')
                )
