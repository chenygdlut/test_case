import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


@dataclass
class KnowledgeEntry:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    relevance_score: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: str = "manual"

    @classmethod
    def create(cls, content: str, metadata: Dict[str, Any] = None, tags: List[str] = None, source: str = "manual"):
        entry_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        return cls(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            source=source
        )


class KnowledgeBase:
    def __init__(self, storage_path: str = "knowledge_base.json"):
        self.storage_path = storage_path
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.tag_index: Dict[str, set] = {}
        self.query_history: List[Dict] = []
        self._load()

    def _load(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.entries = {
                    k: KnowledgeEntry(**v) for k, v in data.get('entries', {}).items()
                }
                self.tag_index = {
                    k: set(v) for k, v in data.get('tag_index', {}).items()
                }
                self.query_history = data.get('query_history', [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({
                'entries': {k: v.__dict__ for k, v in self.entries.items()},
                'tag_index': {k: list(v) for k, v in self.tag_index.items()},
                'query_history': self.query_history[-1000:]
            }, f, ensure_ascii=False, indent=2)

    def add(self, content: str, metadata: Dict[str, Any] = None, 
            tags: List[str] = None, source: str = "manual") -> KnowledgeEntry:
        entry = KnowledgeEntry.create(content, metadata, tags, source)
        self.entries[entry.id] = entry
        
        for tag in entry.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(entry.id)
        
        self._save()
        return entry

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        entry = self.entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now().isoformat()
            self._save()
        return entry

    def search_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        if not tags:
            return list(self.entries.values())
        
        result_ids = None
        for tag in tags:
            if tag in self.tag_index:
                if result_ids is None:
                    result_ids = self.tag_index[tag].copy()
                else:
                    result_ids &= self.tag_index[tag]
        
        if result_ids is None:
            return []
        return [self.entries[eid] for eid in result_ids if eid in self.entries]

    def update_feedback(self, entry_id: str, feedback: FeedbackType, 
                       query_context: str = None):
        if entry_id not in self.entries:
            return
        
        entry = self.entries[entry_id]
        
        if feedback == FeedbackType.POSITIVE:
            entry.success_count += 1
        elif feedback == FeedbackType.NEGATIVE:
            entry.failure_count += 1
        elif feedback == FeedbackType.CORRECTION:
            entry.failure_count += 0.5
        
        total = entry.success_count + entry.failure_count
        if total > 0:
            entry.relevance_score = entry.success_count / total
        
        if query_context:
            self.query_history.append({
                'query': query_context,
                'entry_id': entry_id,
                'feedback': feedback.value,
                'timestamp': datetime.now().isoformat()
            })
        
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        total_entries = len(self.entries)
        total_queries = len(self.query_history)
        
        avg_relevance = sum(e.relevance_score for e in self.entries.values()) / total_entries if total_entries > 0 else 0
        
        return {
            'total_entries': total_entries,
            'total_queries': total_queries,
            'average_relevance': avg_relevance,
            'top_tags': sorted(self.tag_index.keys(), 
                             key=lambda t: len(self.tag_index[t]), 
                             reverse=True)[:10],
            'most_accessed': sorted(self.entries.values(), 
                                   key=lambda e: e.access_count, 
                                   reverse=True)[:5]
        }

    def prune_low_value(self, threshold: float = 0.2):
        to_remove = [
            eid for eid, entry in self.entries.items()
            if entry.relevance_score < threshold and entry.access_count > 3
        ]
        
        for eid in to_remove:
            entry = self.entries[eid]
            for tag in entry.tags:
                if tag in self.tag_index:
                    self.tag_index[tag].discard(eid)
            del self.entries[eid]
        
        if to_remove:
            self._save()
        
        return len(to_remove)
