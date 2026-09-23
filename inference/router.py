import time
import json
from inference.pipeline import DristiPipeline

class DristiRouter:
    def __init__(self):
        # Load our System 1 Engine (Dristi)
        self.pipeline = DristiPipeline()
        
        # We set this to 85% to perfectly demonstrate the routing cutoff.
        # Query 1 (0.89) will pass, Query 2 (0.74) will be routed to System 2.
        self.pipeline.engine.min_confidence = 0.85

    def route_query(self, question: str, options: list[str]) -> dict:
        print(f"\n[Router] Incoming query: '{question}'")
        print("[Router] Phase 1: Sending to Dristi (System 1)...")
        
        # 1. Fast Path: System 1 (Dristi)
        sys1_response = self.pipeline.process(question, options)["final_decision"]
        
        # 2. Did System 1 succeed?
        if not sys1_response.get("abstain", False) and sys1_response.get("decision") != "BLOCKED":
            print("[Router] SUCCESS: System 1 handled it securely and confidently.")
            return {
                "source": "System 1 (Dristi Fast-Path)",
                "decision": sys1_response["decision"],
                "confidence": sys1_response["confidence"],
                "selected_option": sys1_response["selected_option"],
                "estimated_cost": "$0.00",
                "latency": "< 20ms"
            }
            
        # 3. Did the Policy Layer Block it?
        if sys1_response.get("decision") == "BLOCKED":
            print("[Router] BLOCKED: System 1 Policy Layer rejected the query.")
            return {
                "source": "System 1 (Policy Guardrail)", 
                "decision": "BLOCKED",
                "reason": sys1_response.get("policy_flags")
            }
            
        # 4. Slow Path: System 1 Abstained -> Handoff to System 2 (Heavy LLM)
        print("[Router] ABSTAINED: System 1 lacked confidence. Routing to System 2 (Heavy LLM)...")
        sys2_response = self._call_system_2_llm(question, options, sys1_response)
        
        return {
            "source": "System 2 (Heavy LLM Fallback)",
            "decision": sys2_response["decision"],
            "selected_option": sys2_response["selected_option"],
            "estimated_cost": "~$0.02 per call",
            "latency": "~ 1,500ms",
            "handoff_reason": "Low System 1 Confidence / Margin"
        }
        
    def _call_system_2_llm(self, question: str, options: list[str], sys1_context: dict) -> dict:
        """
        This method simulates calling a heavy Generative LLM (like GPT-4, Claude, or Llama-3).
        In a real app, you would use the OpenAI or Anthropic SDK here.
        """
        # We can actually pass Dristi's initial guess to the LLM to give it a head start!
        dristi_guess = sys1_context.get("selected_option")
        
        prompt = f"""
        Question: {question}
        Options: {options}
        Context: A fast classifier guessed '{dristi_guess}', but was uncertain. 
        Please carefully reason through this and provide the definitive answer.
        """
        print(f"  [System 2] Generating prompt for heavy LLM...")
        
        # Simulate the slow latency of a generative LLM
        time.sleep(1.5) 
        
        # Simulate the LLM's final generated answer
        return {
            "decision": "NO", 
            "selected_option": "Unknown" 
        }

if __name__ == "__main__":
    router = DristiRouter()
    
    print("\n==================================================")
    print("      DRISTI COMPOUND AI ROUTER (EDGE 1)")
    print("==================================================")
    
    # Test 1: Easy query (Dristi handles it for free)
    q1 = "Is Python a programming language?"
    opts1 = ["No", "Yes", "Maybe"]
    print(json.dumps(router.route_query(q1, opts1), indent=2))
    
    print("\n" + "-" * 50)
    
    # Test 2: Hard/Ambiguous query (Dristi routes it to GPT-4/Llama)
    q2 = "Does the conceptual colour blue weigh more than forty kilograms in an imaginary universe?"
    opts2 = ["Only on Tuesday", "Exactly 42", "Absolutely", "Unknown"]
    print(json.dumps(router.route_query(q2, opts2), indent=2))
