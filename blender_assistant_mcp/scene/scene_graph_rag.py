import bpy
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Optional
import re

@dataclass
class ObjectSummary:
    """Lightweight representation of a Blender Object for RAG retrieval."""
    name: str
    type: str
    location: tuple
    collection: Optional[str] = None
    materials: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    children: int = 0
    active: bool = False
    selected: bool = False

    def to_text(self) -> str:
        """Convert to a descriptive string for LLM consumption."""
        parts = [f"Name: {self.name}", f"Type: {self.type}"]
        if self.collection:
            parts.append(f"Collection: {self.collection}")
        parts.append(f"Location: {self.location}")
        if self.materials:
            parts.append(f"Materials: {', '.join(self.materials)}")
        if self.modifiers:
            parts.append(f"Modifiers: {', '.join(self.modifiers)}")
        if self.children:
            parts.append(f"Children: {self.children}")
        if self.active:
            parts.append("[ACTIVE]")
        if self.selected:
            parts.append("[SELECTED]")
        return " | ".join(parts)

class SceneGraphIndexer:
    """
    Maintains an searchable index of the current Blender scene.
    Concept adapted from DeepCode's 'CodeIndexer' for efficient memory retrieval.
    """
    _instance = None
    
    def __init__(self):
        self._cache_valid = False
        self._index: Dict[str, ObjectSummary] = {} # Map name -> Summary
        self._keywords: Dict[str, Set[str]] = {} # Map keyword -> Set(Object Names)
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SceneGraphIndexer()
        return cls._instance

    def invalidate(self):
        """Mark index as dirty. Should be called by SceneWatcher."""
        self._cache_valid = False

    def _ensure_valid(self):
        if not self._cache_valid:
            self.rebuild_index()

    def rebuild_index(self):
        """Full scan of bpy.data.objects to build the index."""
        if not bpy.context:
            return 
            
        self._index.clear()
        self._keywords.clear()
        
        active_obj = bpy.context.active_object
        selected_names = {o.name for o in bpy.context.selected_objects}
        
        for obj in bpy.data.objects:
            # 1. Build Summary
            mats = [s.material.name for s in obj.material_slots if s.material]
            mods = [m.name for m in obj.modifiers]
            
            summary = ObjectSummary(
                name=obj.name,
                type=obj.type,
                location=tuple(round(x, 2) for x in obj.location),
                collection=obj.users_collection[0].name if obj.users_collection else None,
                materials=mats,
                modifiers=mods,
                children=len(obj.children),
                active=(obj == active_obj),
                selected=(obj.name in selected_names)
            )
            
            self._index[obj.name] = summary
            
            # 2. Index Keywords (Naive Tokenization)
            text_corpus = f"{obj.name} {obj.type} {' '.join(mats)} {' '.join(mods)}".lower()
            tokens = set(re.findall(r'\w+', text_corpus))
            
            for token in tokens:
                if token not in self._keywords:
                    self._keywords[token] = set()
                self._keywords[token].add(obj.name)
                
        self._cache_valid = True

    def search(self, query: str, limit: int = 20) -> List[ObjectSummary]:
        """
        Search for objects matching the query.
        Returns a list of ObjectSummary.
        """
        self._ensure_valid()
        
        query = query.lower().strip()
        if not query:
            # Return selected or active or top items
            return list(self._index.values())[:limit]
            
        # Naive Search: AND logic for tokens
        query_tokens = set(re.findall(r'\w+', query))
        if not query_tokens:
             return list(self._index.values())[:limit]
             
        candidate_names: Optional[Set[str]] = None
        
        for token in query_tokens:
            # Find exact or partial matches in keyword index
            matches = set()
            for key in self._keywords:
                if token in key: # substring match
                    matches.update(self._keywords[key])
            
            if candidate_names is None:
                candidate_names = matches
            else:
                candidate_names.intersection_update(matches)
                
            if len(candidate_names) == 0:
                break
                
        results = []
        if candidate_names:
            for name in candidate_names:
                if name in self._index:
                    results.append(self._index[name])
                    
        # Sort by relevance (Exact name match > Starts with > Selected > Active)
        def score(s: ObjectSummary):
            sc = 0
            if s.name.lower() == query: sc += 100
            elif s.name.lower().startswith(query): sc += 50
            if s.active: sc += 10
            if s.selected: sc += 5
            return sc
            
        results.sort(key=score, reverse=True)
        return results[:limit]

# Global Accessor
def get_scene_index() -> SceneGraphIndexer:
    return SceneGraphIndexer.get_instance()
