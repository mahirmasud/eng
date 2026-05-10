"""
Domain Loader - Loads domain knowledge from YAML files
Autonomous Recommendation Engine Platform
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class DomainLoader:
    """Load and manage domain knowledge from YAML configuration files."""
    
    def __init__(self, domain_pack_path: Path):
        self.domain_pack_path = Path(domain_pack_path)
        self.cache = {}
    
    def load_all(self) -> Dict[str, Any]:
        """Load all domain knowledge files."""
        knowledge = {
            "keywords": self.load_keywords(),
            "rules": self.load_rules(),
            "schemas": self.load_schemas(),
            "metrics": self.load_metrics(),
            "validations": self.load_validations(),
            "questions": self.load_questions(),
        }
        return knowledge
    
    def load_keywords(self) -> Dict[str, Any]:
        """Load keyword mappings for semantic matching."""
        keywords_file = self.domain_pack_path / "default" / "keywords.yaml"
        
        default_keywords = {
            "target_roles": {
                "USER_ID": "Unique identifier for a user or customer",
                "ITEM_ID": "Unique identifier for an item, product, or content",
                "INTERACTION_EVENT": "Event representing user interaction with item",
                "ENGAGEMENT_SIGNAL": "Signal indicating level of user engagement",
                "TEMPORAL_FEATURE": "Time-related feature for temporal patterns",
                "SESSION_ID": "Identifier for a user session or visit",
                "CONTEXTUAL_FEATURE": "Contextual information about the interaction",
                "ITEM_CATEGORY": "Category or classification of an item",
                "ITEM_TITLE": "Title or name of an item",
                "ITEM_DESCRIPTION": "Description or details of an item",
                "USER_AGE": "Age or age group of a user",
                "USER_GENDER": "Gender of a user",
                "USER_LOCATION": "Geographic location of a user",
                "PRICE": "Price or cost of an item",
                "RATING": "Explicit rating given by a user",
                "CLICK": "Click event or count",
                "VIEW": "View event or count",
                "PURCHASE": "Purchase event or transaction",
                "DWELL_TIME": "Time spent viewing or interacting",
                "IMPRESSION": "Impression or exposure event",
            },
            "synonyms": {
                "user": ["customer", "member", "account", "visitor", "buyer"],
                "item": ["product", "article", "document", "content", "movie", "book", "song"],
                "click": ["tap", "select", "activate"],
                "view": ["see", "look", "display", "show"],
                "purchase": ["buy", "order", "transaction", "checkout"],
                "rating": ["score", "review", "feedback", "stars"],
                "time": ["duration", "length", "period", "span"],
            },
            "patterns": {
                "id_patterns": ["id", "uuid", "_id", "identifier", "key"],
                "temporal_patterns": ["date", "time", "timestamp", "created", "updated", "at"],
                "numeric_patterns": ["count", "amount", "value", "score", "price"],
                "categorical_patterns": ["type", "category", "class", "group", "status"],
            },
        }
        
        if keywords_file.exists():
            with open(keywords_file, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    # Merge with defaults
                    for key in loaded:
                        if key in default_keywords:
                            if isinstance(default_keywords[key], dict):
                                default_keywords[key].update(loaded[key])
                            else:
                                default_keywords[key] = loaded[key]
                        else:
                            default_keywords[key] = loaded[key]
        
        self.cache["keywords"] = default_keywords
        return default_keywords
    
    def load_rules(self) -> Dict[str, Any]:
        """Load boosting and mapping rules."""
        rules_file = self.domain_pack_path / "default" / "rules.yaml"
        
        default_rules = {
            "boosting_rules": [
                {"pattern": "user.*id", "target": "USER_ID", "boost": 0.3},
                {"pattern": "item.*id", "target": "ITEM_ID", "boost": 0.3},
                {"pattern": ".*click.*", "target": "CLICK", "boost": 0.2},
                {"pattern": ".*view.*", "target": "VIEW", "boost": 0.2},
                {"pattern": ".*purchase.*", "target": "PURCHASE", "boost": 0.2},
                {"pattern": ".*rating.*", "target": "RATING", "boost": 0.2},
                {"pattern": ".*timestamp.*", "target": "TEMPORAL_FEATURE", "boost": 0.1},
            ],
            "exclusion_rules": [
                {"pattern": ".*password.*", "exclude": True},
                {"pattern": ".*email.*", "exclude": True},
                {"pattern": ".*token.*", "exclude": True},
            ],
            "priority_rules": [
                {"target": "USER_ID", "priority": 1},
                {"target": "ITEM_ID", "priority": 1},
                {"target": "INTERACTION_EVENT", "priority": 2},
            ],
        }
        
        if rules_file.exists():
            with open(rules_file, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    for key in loaded:
                        if key in default_rules and isinstance(default_rules[key], list):
                            default_rules[key].extend(loaded[key])
                        else:
                            default_rules[key] = loaded[key]
        
        self.cache["rules"] = default_rules
        return default_rules
    
    def load_schemas(self) -> Dict[str, Any]:
        """Load schema definitions for common domains."""
        schema_file = self.domain_pack_path / "default" / "schema.yaml"
        
        default_schemas = {
            "ecommerce": {
                "required": ["user_id", "item_id", "event_type", "timestamp"],
                "optional": ["session_id", "price", "quantity", "category"],
            },
            "media": {
                "required": ["user_id", "content_id", "action", "timestamp"],
                "optional": ["duration", "completion_rate", "device"],
            },
            "news": {
                "required": ["user_id", "article_id", "event", "timestamp"],
                "optional": ["section", "author", "read_time"],
            },
        }
        
        if schema_file.exists():
            with open(schema_file, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_schemas.update(loaded)
        
        self.cache["schemas"] = default_schemas
        return default_schemas
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load evaluation metrics configuration."""
        metrics_file = self.domain_pack_path / "default" / "metrics.yaml"
        
        default_metrics = {
            "retrieval": ["recall@k", "precision@k", "map", "mrr"],
            "ranking": ["ndcg@k", "dcg@k", "hit_rate"],
            "business": ["ctr", "conversion_rate", "revenue_per_user"],
        }
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_metrics.update(loaded)
        
        self.cache["metrics"] = default_metrics
        return default_metrics
    
    def load_validations(self) -> Dict[str, Any]:
        """Load validation rules."""
        validations_file = self.domain_pack_path / "default" / "validations.yaml"
        
        default_validations = {
            "data_quality": {
                "min_rows": 1000,
                "max_null_ratio": 0.5,
                "require_unique_ids": True,
            },
            "schema_checks": {
                "require_timestamps": True,
                "require_positive_values": ["price", "quantity", "rating"],
            },
        }
        
        if validations_file.exists():
            with open(validations_file, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_validations.update(loaded)
        
        self.cache["validations"] = default_validations
        return default_validations
    
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load human-in-the-loop questions for ambiguous cases."""
        questions_file = self.domain_pack_path / "default" / "questions.yaml"
        
        default_questions = [
            {
                "condition": "confidence < 0.6",
                "question": "Column '{column}' detected as {role} with {confidence:.0%} confidence. Is this correct?",
                "options": ["Yes", "No", "Unsure"],
            },
            {
                "condition": "multiple_matches",
                "question": "Column '{column}' could be {roles}. Which is most appropriate?",
                "options": "dynamic",
            },
        ]
        
        if questions_file.exists():
            with open(questions_file, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_questions.extend(loaded)
        
        self.cache["questions"] = default_questions
        return default_questions
    
    def get_keyword_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word."""
        keywords = self.load_keywords()
        synonyms = keywords.get("synonyms", {})
        
        result = [word]
        for key, syns in synonyms.items():
            if word.lower() in key.lower() or key.lower() in word.lower():
                result.extend(syns)
        
        return result
    
    def get_boost_rule(self, column: str, target: str) -> float:
        """Get boost value for a column-target pair."""
        rules = self.load_rules()
        boosting_rules = rules.get("boosting_rules", [])
        
        import re
        for rule in boosting_rules:
            pattern = rule.get("pattern", "")
            if rule.get("target") == target and re.match(pattern, column, re.IGNORECASE):
                return rule.get("boost", 0.0)
        
        return 0.0
