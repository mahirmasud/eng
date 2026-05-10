"""
Logic Booster - YAML-driven boosting and synonym weighting
Autonomous Recommendation Engine Platform
"""

import re
from typing import Dict, List, Any, Optional


class LogicBooster:
    """
    Apply logic-based boosting to semantic matches.
    
    Features:
    - Pattern-based boost rules
    - Synonym weighting
    - Context-aware scoring
    - Domain rule injection
    """
    
    def __init__(self):
        self.boost_rules = []
        self.synonyms = {}
        self.exclusion_patterns = []
        self.priority_rules = {}
        
        # Default synonyms for common terms
        self._init_default_synonyms()
    
    def _init_default_synonyms(self):
        """Initialize default synonym mappings."""
        self.synonyms = {
            "user": ["customer", "member", "account", "visitor", "buyer", "client"],
            "item": ["product", "article", "document", "content", "movie", "book", "song", "video"],
            "click": ["tap", "select", "activate", "press"],
            "view": ["see", "look", "display", "show", "watch", "read"],
            "purchase": ["buy", "order", "transaction", "checkout", "acquire"],
            "rating": ["score", "review", "feedback", "stars", "vote"],
            "time": ["duration", "length", "period", "span", "elapsed"],
            "id": ["identifier", "uuid", "key", "code"],
            "category": ["type", "class", "group", "section", "genre"],
        }
    
    def load_rules(self, rules: Dict[str, Any]) -> None:
        """Load boosting rules from configuration."""
        self.boost_rules = rules.get("boosting_rules", [])
        self.exclusion_patterns = rules.get("exclusion_rules", [])
        self.priority_rules = {
            r["target"]: r["priority"] 
            for r in rules.get("priority_rules", [])
        }
    
    def get_boost(
        self, 
        column: str, 
        target_role: str
    ) -> float:
        """Get boost factor for a column-target pair."""
        total_boost = 0.0
        reasons = []
        
        # Check pattern-based rules
        for rule in self.boost_rules:
            pattern = rule.get("pattern", "")
            if rule.get("target") == target_role:
                if re.match(pattern, column, re.IGNORECASE):
                    boost = rule.get("boost", 0.0)
                    total_boost += boost
                    reasons.append(f"pattern:{pattern}")
        
        # Check synonym-based boosts
        synonym_boost = self._get_synonym_boost(column, target_role)
        if synonym_boost > 0:
            total_boost += synonym_boost
            reasons.append("synonym_match")
        
        # Check priority-based boosts
        if target_role in self.priority_rules:
            priority = self.priority_rules[target_role]
            if priority == 1:  # Highest priority
                total_boost += 0.1
                reasons.append("high_priority")
        
        return total_boost
    
    def _get_synonym_boost(
        self, 
        column: str, 
        target_role: str
    ) -> float:
        """Calculate boost based on synonym matches."""
        column_lower = column.lower()
        boost = 0.0
        
        # Map target roles to synonym keys
        role_to_synonym = {
            "USER_ID": "user",
            "ITEM_ID": "item",
            "CLICK": "click",
            "VIEW": "view",
            "PURCHASE": "purchase",
            "RATING": "rating",
            "TEMPORAL_FEATURE": "time",
            "ITEM_CATEGORY": "category",
        }
        
        synonym_key = role_to_synonym.get(target_role)
        if synonym_key and synonym_key in self.synonyms:
            synonyms = self.synonyms[synonym_key]
            # Check if any synonym appears in column name
            for syn in synonyms:
                if syn in column_lower:
                    boost += 0.05  # Small boost per synonym match
            
            boost = min(boost, 0.2)  # Cap at 0.2
        
        return boost
    
    def get_boost_reasons(
        self, 
        column: str, 
        target_role: str
    ) -> List[str]:
        """Get list of reasons for applied boosts."""
        reasons = []
        
        for rule in self.boost_rules:
            pattern = rule.get("pattern", "")
            if rule.get("target") == target_role:
                if re.match(pattern, column, re.IGNORECASE):
                    reasons.append(f"rule:{pattern}")
        
        # Check synonyms
        column_lower = column.lower()
        for key, syns in self.synonyms.items():
            for syn in syns:
                if syn in column_lower:
                    reasons.append(f"synonym:{syn}")
        
        return reasons
    
    def is_excluded(self, column: str) -> bool:
        """Check if column should be excluded from mapping."""
        column_lower = column.lower()
        
        for rule in self.exclusion_patterns:
            pattern = rule.get("pattern", "")
            if re.match(pattern, column_lower, re.IGNORECASE):
                return rule.get("exclude", True)
        
        return False
    
    def apply_context_boost(
        self,
        column: str,
        target_role: str,
        context: Dict[str, Any]
    ) -> float:
        """Apply context-aware boosting."""
        boost = 0.0
        
        # Dataset-level context
        dataset_type = context.get("dataset_type", "")
        
        # E-commerce specific boosts
        if dataset_type == "ecommerce":
            if target_role == "PURCHASE" and "price" in column.lower():
                boost += 0.1
            if target_role == "ITEM_ID" and "product" in column.lower():
                boost += 0.1
        
        # Media specific boosts
        if dataset_type == "media":
            if target_role == "VIEW" and "watch" in column.lower():
                boost += 0.1
            if target_role == "DWELL_TIME" and "duration" in column.lower():
                boost += 0.1
        
        return boost
    
    def get_priority(self, target_role: str) -> int:
        """Get priority level for a target role."""
        return self.priority_rules.get(target_role, 999)
    
    def add_synonym(self, key: str, synonyms: List[str]) -> None:
        """Add new synonyms to the booster."""
        if key not in self.synonyms:
            self.synonyms[key] = []
        self.synonyms[key].extend(synonyms)
    
    def add_boost_rule(
        self, 
        pattern: str, 
        target: str, 
        boost: float
    ) -> None:
        """Add a new boost rule."""
        self.boost_rules.append({
            "pattern": pattern,
            "target": target,
            "boost": boost,
        })
