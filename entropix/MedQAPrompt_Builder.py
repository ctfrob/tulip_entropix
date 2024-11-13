import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MedicalQuestion:
    """Data class for medical question structure"""
    question: str
    answer: str  # Text form of answer
    options: Dict[str, str]
    meta_info: str
    answer_idx: str  # Letter index (A, B, C, etc.)

class MedicalQAPrompt:
    """
    Handles system prompts and questions for medical QA using USMLE-style format.
    """
    
    SYSTEM_PROMPT = """You are an expert medical AI assistant specializing in USMLE-style multiple choice questions. You combine systematic clinical reasoning with precise medical knowledge to select the single best answer from the provided options.

QUESTION ANALYSIS
1. Question Framework:
   - Identify question type:
     * Direct knowledge/recall
     * Patient case scenario
     * Lab/imaging interpretation
     * Basic science application
     * Management/treatment decision
   - Note specific task (diagnosis, next step, mechanism, etc.)
   - Identify key testing concept

2. Evidence Integration:
   - For any provided evidence/references:
     * Identify most relevant passages
     * Note relationships between different evidence pieces
     * Recognize gaps requiring medical knowledge
     * Connect evidence to clinical scenario

CLINICAL REASONING STRUCTURE
For Patient Cases:
1. Systematic Information Gathering:
   - Demographics and context
   - Timing and progression of symptoms
   - Relevant history (medical, social, family)
   - Physical examination findings
   - Laboratory/imaging results
   - Treatments and responses

2. Clinical Analysis:
   - Pattern recognition
   - Risk factor evaluation
   - Symptom clustering
   - Timeline integration
   - Severity assessment

3. Diagnostic Reasoning:
   - Generate initial differential
   - Apply epidemiology
   - Consider mechanism
   - Evaluate fit with presentation

For Knowledge Questions:
1. Core Concept Analysis:
   - Identify fundamental principle
   - Note relevant systems/pathways
   - Consider clinical applications
   - Recognize related concepts

2. Knowledge Integration:
   - Connect basic and clinical science
   - Apply pathophysiology
   - Consider mechanism of action
   - Link to clinical relevance

MULTIPLE CHOICE STRATEGY
1. Pre-option Analysis:
   - Form expected answer
   - List discriminating features
   - Note critical findings

2. Option Evaluation:
   - Consider each option independently
   - Identify distractor patterns:
     * Right concept, wrong timing
     * Associated but incorrect
     * Common misconceptions
     * Partially correct but incomplete

3. Comparative Analysis:
   - Rank all options
   - Test against clinical scenario
   - Apply elimination strategy
   - Verify best fit

RESPONSE STRUCTURE
1. Answer Declaration:
   - State selected option clearly
   - Express confidence level
   - Note key assumptions

2. Reasoning Explanation:
   - Present logical flow
   - Link evidence to conclusion
   - Explain clinical relevance
   - Address key discriminators

3. Alternative Discussion:
   - Explain why other options incorrect
   - Address tempting distractors
   - Note close alternatives
   - Clarify subtle distinctions

QUALITY CHECKS
- Verify single best answer
- Confirm within scope of question
- Use precise medical terminology
- Maintain clinical relevance
- Check reasoning consistency
- Confirm evidence support

Remember:
- Always select ONE best answer
- Stay within provided information
- Use systematic approach
- Show clear reasoning steps
- Link to clinical significance
- Address all major alternatives
"""
    
    def format_system_prompt(self) -> str:
        """Format system prompt with required tokens"""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.SYSTEM_PROMPT}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def format_user_prompt(self, question: MedicalQuestion) -> str:
        """Format user question with required tokens"""
        formatted_question = self.format_question(question)
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{formatted_question}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def format_question(self, question: MedicalQuestion) -> str:
        """Format the question and options for presentation"""
        options_text = "\n".join([f"{key}: {value}" for key, value in sorted(question.options.items())])
        
        return (
            f"Question Type: {question.meta_info}\n\n"
            f"Question:\n{question.question}\n\n"
            f"Options:\n{options_text}"
        )

    def format_output(self, question: MedicalQuestion, model_output: str) -> str:
        """Format question and model output for display"""
        return (
            "\n" + "="*80 + "\n"
            f"QUESTION TYPE: {question.meta_info}\n"
            f"QUESTION: {question.question}\n\n"
            "OPTIONS:\n" + 
            "\n".join([f"{key}: {value}" for key, value in sorted(question.options.items())]) + 
            "\n\n"
            f"CORRECT ANSWER: {question.answer} ({question.answer_idx})\n\n"
            f"MODEL OUTPUT:\n{model_output}\n"
            "="*80 + "\n"
        )

    @staticmethod
    def parse_question_json(json_str: str) -> MedicalQuestion:
        """Parse a JSON string into a MedicalQuestion object"""
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        return MedicalQuestion(
            question=data["question"],
            answer=data["answer"],
            options=data["options"],
            meta_info=data["meta_info"],
            answer_idx=data["answer_idx"]
        )

    def get_prompt(self, question_json: str) -> Tuple[str, MedicalQuestion]:
        """Generate the complete formatted prompt"""
        question = self.parse_question_json(question_json)
        return self.format_user_prompt(question), question

    def process_jsonl_file(self, file_path: str, max_questions: Optional[int] = None) -> List[Tuple[str, MedicalQuestion]]:
        """Process multiple questions from a JSONL file"""
        prompts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if max_questions and i > max_questions:
                        break
                        
                    if line.strip():
                        try:
                            prompt, question = self.get_prompt(line)
                            prompts.append((prompt, question))
                        except Exception as e:
                            print(f"Error processing question {i}: {e}")
            
            return prompts
            
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

def main():
    """Example usage"""
    # Just for testing prompt generation
    prompt_handler = MedicalQAPrompt()
    
    # Example single question processing
    example_question = {
        "question": "A 4-year-old boy is brought to the emergency department...",
        "answer": "Perform emergency laparotomy",
        "options": {
            "A": "Get consent from the patient's brother",
            "B": "Get consent from the patient",
            "C": "Obtain a court order for surgery",
            "D": "Schedule hospital ethics consult",
            "E": "Perform emergency laparotomy",
            "F": "Delay surgery until parental consent"
        },
        "meta_info": "step2",
        "answer_idx": "E"
    }
    
    prompt, question = prompt_handler.get_prompt(example_question)
    print("Generated prompt:")
    print(prompt)
    print("\nFormatted output example:")
    print(prompt_handler.format_output(question, "Sample model output text here"))

if __name__ == "__main__":
    main()