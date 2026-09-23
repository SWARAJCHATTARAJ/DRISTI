class PolicyLayer:
    def __init__(self):
        # A simple list of blocked keywords to demonstrate deterministic overrides
        self.blocked_keywords = ["harmful", "illegal", "unethical"]

    def apply_policy(self, user_question: str, model_output: dict) -> dict:
        """
        Takes the raw inference output and applies deterministic business/safety rules.
        """
        final_result = model_output.copy()
        final_result["policy_flags"] = []
        
        # 1. Blocked Keyword Check
        question_lower = user_question.lower()
        if any(kw in question_lower for kw in self.blocked_keywords):
            final_result["decision"] = "BLOCKED"
            final_result["score"] = 0
            final_result["abstain"] = True
            final_result["policy_flags"].append("blocked_keyword_detected")
            return final_result

        # 2. Handle Model Abstention
        if model_output.get("abstain", False):
            final_result["policy_flags"].append("model_abstained_due_to_uncertainty")
            # We might want to clear the selected option if we are abstaining
            # final_result["selected_option"] = None
        
        # 3. Score Overrides based on Decision
        if model_output.get("decision") == "NO" and model_output.get("score", 0) > 3:
            final_result["policy_flags"].append("score_capped_for_no_decision")
            final_result["score"] = min(3, model_output.get("score"))
            
        return final_result
