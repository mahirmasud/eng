"""
Data Validator
Autonomous Recommendation Engine Platform
"""

from typing import Dict, List, Any
from datetime import datetime


class DataValidator:
    """Validate datasets against rules and constraints."""
    
    def __init__(self):
        self.default_rules = {
            "min_rows": 100,
            "max_null_percentage": 50.0,
            "require_user_id": True,
            "require_item_id": True,
        }
    
    def validate(
        self,
        file_info: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a dataset profile against rules."""
        
        issues = []
        warnings = []
        passed_checks = []
        
        # Check minimum rows
        if profile["num_rows"] < self.default_rules["min_rows"]:
            issues.append({
                "type": "ERROR",
                "code": "INSUFFICIENT_ROWS",
                "message": f"Dataset has {profile['num_rows']} rows, minimum is {self.default_rules['min_rows']}",
            })
        else:
            passed_checks.append("MIN_ROWS_CHECK")
        
        # Check null percentage
        null_pct = profile["data_quality"]["null_percentage"]
        if null_pct > self.default_rules["max_null_percentage"]:
            issues.append({
                "type": "WARNING",
                "code": "HIGH_NULL_PERCENTAGE",
                "message": f"Dataset has {null_pct}% null values",
            })
        else:
            passed_checks.append("NULL_PERCENTAGE_CHECK")
        
        # Check for user ID column
        columns = [c["name"].lower() for c in profile["columns"]]
        has_user_id = any("user" in c or "uid" in c for c in columns)
        has_item_id = any("item" in c or "product" in c for c in columns)
        
        if self.default_rules["require_user_id"] and not has_user_id:
            warnings.append({
                "type": "WARNING",
                "code": "NO_USER_ID",
                "message": "No obvious user ID column detected",
            })
        else:
            passed_checks.append("USER_ID_CHECK")
        
        if self.default_rules["require_item_id"] and not has_item_id:
            warnings.append({
                "type": "WARNING",
                "code": "NO_ITEM_ID",
                "message": "No obvious item ID column detected",
            })
        else:
            passed_checks.append("ITEM_ID_CHECK")
        
        # Check for interaction signal
        interaction_patterns = ["click", "view", "purchase", "rating", "score"]
        has_interaction = any(
            any(p in c for p in interaction_patterns) 
            for c in columns
        )
        
        if not has_interaction:
            warnings.append({
                "type": "INFO",
                "code": "NO_INTERACTION_SIGNAL",
                "message": "No obvious interaction signal column detected",
            })
        else:
            passed_checks.append("INTERACTION_CHECK")
        
        validation_result = {
            "dataset_name": profile["dataset_name"],
            "validated_at": datetime.now().isoformat(),
            "is_valid": len([i for i in issues if i["type"] == "ERROR"]) == 0,
            "issues": issues,
            "warnings": warnings,
            "passed_checks": passed_checks,
            "summary": {
                "total_issues": len(issues),
                "total_warnings": len(warnings),
                "passed_checks_count": len(passed_checks),
            },
        }
        
        return validation_result
    
    def add_rule(self, rule_name: str, rule_value: Any):
        """Add or update a validation rule."""
        self.default_rules[rule_name] = rule_value
    
    def get_rules(self) -> Dict[str, Any]:
        """Get all validation rules."""
        return self.default_rules.copy()
