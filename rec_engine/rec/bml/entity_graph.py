"""
Entity Graph Builder - Builds entity relationship graphs
Autonomous Recommendation Engine Platform
"""

from typing import Dict, List, Any


class EntityGraphBuilder:
    """Build entity relationship graphs for recommendation systems."""
    
    def build(
        self,
        users: List[str],
        items: List[str],
        interactions: List[str],
        sessions: List[str],
        features: List[str],
    ) -> Dict[str, Any]:
        """Build an entity graph from classified components."""
        
        # Build nodes
        nodes = {
            "user_entities": [
                {"name": col, "type": "user", "is_primary": i == 0}
                for i, col in enumerate(users)
            ],
            "item_entities": [
                {"name": col, "type": "item", "is_primary": i == 0}
                for i, col in enumerate(items)
            ],
            "interaction_entities": [
                {"name": col, "type": "interaction"}
                for col in interactions
            ],
            "session_entities": [
                {"name": col, "type": "session"}
                for col in sessions
            ],
            "feature_entities": [
                {"name": col, "type": "feature"}
                for col in features
            ],
        }
        
        # Build edges (relationships)
        edges = []
        
        # User-Item relationships through interactions
        primary_user = next((u["name"] for u in nodes["user_entities"] if u.get("is_primary")), None)
        primary_item = next((i["name"] for i in nodes["item_entities"] if i.get("is_primary")), None)
        
        if primary_user and primary_item:
            for interaction in interactions:
                edges.append({
                    "source": primary_user,
                    "target": primary_item,
                    "via": interaction,
                    "type": "interaction",
                })
            
            # Session relationship
            for session in sessions:
                edges.append({
                    "source": primary_user,
                    "target": session,
                    "type": "belongs_to",
                })
        
        # Feature relationships
        for feature in features:
            if any(u in feature.lower() for u in ["user", "customer", "member"]):
                edges.append({
                    "source": primary_user,
                    "target": feature,
                    "type": "has_feature",
                })
            elif any(i in feature.lower() for i in ["item", "product", "content"]):
                edges.append({
                    "source": primary_item,
                    "target": feature,
                    "type": "has_feature",
                })
        
        # Summary statistics
        summary = {
            "total_nodes": sum(len(v) for v in nodes.values()),
            "total_edges": len(edges),
            "user_count": len(users),
            "item_count": len(items),
            "interaction_count": len(interactions),
            "session_count": len(sessions),
            "feature_count": len(features),
        }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary,
            "primary_user": primary_user,
            "primary_item": primary_item,
        }
