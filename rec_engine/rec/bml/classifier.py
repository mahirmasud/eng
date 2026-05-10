"""
Role Classifier - Classifies columns into business roles
Autonomous Recommendation Engine Platform
"""

from typing import Dict, List, Any


class RoleClassifier:
    """Classify semantic roles into business categories."""
    
    # Classification mappings
    ROLE_TO_CLASSIFICATION = {
        "USER_ID": "IDENTITY",
        "ITEM_ID": "IDENTITY",
        "CLICK": "SIGNAL",
        "VIEW": "SIGNAL",
        "PURCHASE": "SIGNAL",
        "RATING": "SIGNAL",
        "ENGAGEMENT_SIGNAL": "SIGNAL",
        "IMPRESSION": "SIGNAL",
        "INTERACTION_EVENT": "EVENT",
        "SESSION_ID": "SESSION",
        "TEMPORAL_FEATURE": "TEMPORAL",
        "ITEM_CATEGORY": "FEATURE",
        "ITEM_TITLE": "FEATURE",
        "ITEM_DESCRIPTION": "FEATURE",
        "USER_AGE": "FEATURE",
        "USER_GENDER": "FEATURE",
        "USER_LOCATION": "FEATURE",
        "PRICE": "FEATURE",
        "DWELL_TIME": "FEATURE",
        "CONTEXTUAL_FEATURE": "CONTEXTUAL",
    }
    
    # Side classification (user, item, interaction)
    ROLE_TO_SIDE = {
        "USER_ID": "user",
        "USER_AGE": "user",
        "USER_GENDER": "user",
        "USER_LOCATION": "user",
        "ITEM_ID": "item",
        "ITEM_CATEGORY": "item",
        "ITEM_TITLE": "item",
        "ITEM_DESCRIPTION": "item",
        "PRICE": "item",
        "CLICK": "interaction",
        "VIEW": "interaction",
        "PURCHASE": "interaction",
        "RATING": "interaction",
        "ENGAGEMENT_SIGNAL": "interaction",
        "IMPRESSION": "interaction",
        "INTERACTION_EVENT": "interaction",
        "SESSION_ID": "session",
        "TEMPORAL_FEATURE": "context",
        "DWELL_TIME": "interaction",
        "CONTEXTUAL_FEATURE": "context",
    }
    
    def classify_target(self, target_role: str) -> str:
        """Get classification for a target role."""
        return self.ROLE_TO_CLASSIFICATION.get(target_role, "FEATURE")
    
    def determine_side(self, target_role: str) -> str:
        """Determine which side of the recommendation system a role belongs to."""
        return self.ROLE_TO_SIDE.get(target_role, "unknown")
    
    def is_identity(self, target_role: str) -> bool:
        """Check if role is an identity field."""
        return self.classify_target(target_role) == "IDENTITY"
    
    def is_signal(self, target_role: str) -> bool:
        """Check if role is a signal/engagement field."""
        return self.classify_target(target_role) == "SIGNAL"
    
    def is_feature(self, target_role: str) -> bool:
        """Check if role is a feature field."""
        return self.classify_target(target_role) == "FEATURE"
    
    def get_all_identities(self) -> List[str]:
        """Get all identity roles."""
        return [
            role for role, cls in self.ROLE_TO_CLASSIFICATION.items()
            if cls == "IDENTITY"
        ]
    
    def get_all_signals(self) -> List[str]:
        """Get all signal roles."""
        return [
            role for role, cls in self.ROLE_TO_CLASSIFICATION.items()
            if cls == "SIGNAL"
        ]
    
    def get_priority_order(self) -> List[str]:
        """Get roles in priority order for mapping."""
        return [
            "USER_ID",
            "ITEM_ID", 
            "INTERACTION_EVENT",
            "ENGAGEMENT_SIGNAL",
            "SESSION_ID",
            "TEMPORAL_FEATURE",
            "CONTEXTUAL_FEATURE",
            "ITEM_CATEGORY",
            "ITEM_TITLE",
            "ITEM_DESCRIPTION",
            "USER_AGE",
            "USER_GENDER",
            "USER_LOCATION",
            "PRICE",
            "RATING",
            "CLICK",
            "VIEW",
            "PURCHASE",
            "DWELL_TIME",
            "IMPRESSION",
        ]
