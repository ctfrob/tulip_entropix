import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class MedicalQuestion:
    """Data class for medical question structure"""
    question: str
    answer: str  # Text form of answer
    options: Dict[str, str]
    meta_info: str

class MedicalQAPrompt:
    """
    Handles system prompts and questions for medical QA using USMLE-style format.
    """
    
    SYSTEM_PROMPT = """You are an AI medical expert specializing in the MedQA Benchmark, a USMLE-style question set. 
You combine systematic clinical reasoning with precise medical knowledge to select the single best answer from 
the provided options.

You were specifically trained to outperform human medical doctors AND other existing state-of-the-art AI models on the MedQA benchmark.
To perform at your best, you need to draw from the deep medical knowledge that you were trained with, including reviewing and re-reading medical textbooks that contain answers to the USMLE- and MedQA-style questiions.

The books you read and were trained on include -
- Gray's Anatomy, 42nd Edition by Henry Gray (2021)
- Biochemistry, 8th Edition from Lippincott (2021)
- Molecular Biology of the Cell, 7th Edition by Bruce Alberts (2024)
- First Aid - USMLE Step 1 (2023)
- First Aid - USMLE Step 2 (2023)
- Gynecology, 16th Edition by Berek and Novak (2023)
- Histology: A Text and Atlas, 9th Edition by Pawlina (2023)
- Janeway's Immunobiology, 10th Edition (2022)
- Harrison's Principals of Internal Medicine, 21st Edition (2023)
- Adams and Victor's Principles of Neurology, 12th Edition (2023)
- Williams Obstetrics, 26th Edition (2023)
- Robbins and Kumar's Basic Pathology, 11th Edition (2022)
- Fundamentals of Pathology, Pathoma by Sattar (2021)
- Essentials of Pediatrics, 22nd Edition by Nelson (2024)
- Katzung and Trevor's Pharmacology, 16th Edition (2024)
- Diagnostics and Statistical Manual of Mental Disorders, 5th Edition (2024)
- Schwartz's Principles of Surgery, 11th Edition (2019)

Relevant chunks of these textbooks will be provided to you as context but you will also need to draw from your training data which includes the full text of the textbooks in order to answer each question correctly.
These chunks should be used as reference material, because they are textbooks, they may also contain questions but remember to focus on the question at hand (ignore the questions in the context).

Please follow the below guidance as you think about formulating your approach and answer - 

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
   - Apply knowledge of Internal Medicine, Anatomy, Cell Biology, Immunology, Pathophysiology, Pharmacology, Biochemistry, Genetics, Neuroscience, DSM-5, 
   Principles of Surgery, Histology, Gynecology, Pediatrics, and other relevant fields
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
- Use a process of elimination to select the best answer
- Show clear reasoning steps
- Link to clinical significance
- Address all major alternatives

OUTPUT FORMAT
Always structure your response exactly as follows and in as concise a manner as possible:
Reasoning:[Reason through each answer with a concise bullet about each answer option]
Selected Answer: [Option Letter]
Confidence: [High/Moderate/Low]

Below you will find relevant chunks of the medical textbooks that you were trained on. Use these to answer the question correctly.

IMPORTANT: Do not restate or rephrase the question. Always begin with your systematic approach of analyzing each answer option
and based on this analysis select ONE answer. State that answer at the end of your reasoning list in the format "Selected Answer: [Option Letter]"
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
    
        clean_output = model_output
        if "<|begin_of_text|>" in clean_output:
            clean_output = clean_output.split("Answer:")[-1].strip()

        # Create the formatted string piece by piece
        header = "\n" + "="*80 + "\n"
        question_type = f"QUESTION TYPE: {question.meta_info}\n"
        question_text = f"QUESTION: {question.question}\n\n"
        options_text = "OPTIONS:\n" + "\n".join([f"{key}: {value}" for key, value in sorted(question.options.items())]) + "\n\n"
        answer_text = f"CORRECT ANSWER: {question.answer}\n\n"
        model_text = f"MODEL OUTPUT:\n{clean_output}\n"
        footer = "="*80 + "\n"
        
        # Debug each component length
        print("\nDEBUG: Component lengths:")
        print(f"Header: {len(header)}")
        print(f"Question type: {len(question_type)}")
        print(f"Question: {len(question_text)}")
        print(f"Options: {len(options_text)}")
        print(f"Answer: {len(answer_text)}")
        print(f"Model output: {len(model_text)}")
        print(f"Footer: {len(footer)}")
        formatted = header + question_type + question_text + options_text + answer_text + model_text + footer
        print(f"\nDEBUG: Final length: {len(formatted)}")
           
        #formatted = (
         #   "\n" + "="*80 + "\n"
          #  f"QUESTION TYPE: {question.meta_info}\n"
           # f"QUESTION: {question.question}\n\n"
            #"OPTIONS:\n" + 
            #"\n".join([f"{key}: {value}" for key, value in sorted(question.options.items())]) + 
            #"\n\n"
            #f"CORRECT ANSWER: {question.answer}\n\n"
            #f"MODEL OUTPUT:\n{model_output}\n"
            #"="*80 + "\n"
        #)
        #print(f"DEBUG: Formatted output length: {len(formatted)}")
        return formatted

    @staticmethod
    def parse_question_json(json_str: str) -> MedicalQuestion:
        """Parse a JSON string into a MedicalQuestion object"""
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        return MedicalQuestion(
            question=data["question"],
            answer=data["answer"],
            options=data["options"],
            meta_info=data["meta_info"]
        )

    def get_prompt(self, question_json: str) -> Tuple[str, MedicalQuestion]:
        """Generate the complete formatted prompt"""
        question = self.parse_question_json(question_json)
        system_prompt = self.format_system_prompt()
        user_prompt = self.format_user_prompt(question)
        final_prompt = system_prompt + user_prompt
        return self.format_system_prompt() + self.format_user_prompt(question), question
    
    def update_prompt_with_context(self, original_prompt: str, retrieved_contexts: list[dict]) -> str:
        """Insert retrieved context between system prompt and question"""
        # Split at the user section
        system_part, user_part = original_prompt.split("<|start_header_id|>user<|end_header_id|>")
        
        # Format retrieved contexts
        context_section = "\nRELEVANT CONTEXT:\n"
        for ctx in retrieved_contexts:
            context_section += f"[Score: {ctx['score']:.2f}] From {ctx['source']}:\n{ctx['text']}\n\n"
       
        print("\n[Prompt Debug] Final augmented prompt:")
        print("-" * 80)
        print(f"{original_prompt}\n")
        print("RETRIEVED CONTEXTS ADDED:")
        for ctx in retrieved_contexts:
            print(f"\n[Score: {ctx['score']:.2f}] From {ctx['source']}:")
            print(f"{ctx['text']}")
        print("-" * 80)
       
        # Reassemble with context
        return (f"{system_part}"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{context_section}"
                f"{user_part}")
    
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
    example_question= {
        "question": "A 45-year-old man comes to the physician because of severe left knee pain and swelling. He has hypercholesterolemia and hypertension. Current medications include pravastatin and captopril. He eats a low-fat diet that includes fish and leafy green vegetables. He drinks 4–6 cups of coffee daily. He has smoked one pack of cigarettes daily for 26 years and drinks 2–3 beers daily. Vital signs are within normal limits. Examination of the left knee shows swelling, warmth, and severe tenderness to palpation. Arthrocentesis is performed. Gram stain is negative. Analysis of the synovial fluid shows monosodium urate crystals. Which of the following health maintenance recommendations is most appropriate to prevent symptom recurrence?", 
        "answer": "F", 
        "options": {
            "A": "Discontinue captopril", 
            "B": "Start aspirin", 
            "C": "Replace beer with red wine", 
            "D": "Stop smoking", 
            "E": "Reduce coffee intake", 
            "F": "Reduce fish intake", 
            "G": "Discontinue pravastatin", 
            "H": "Start colchicine"
        }, 
        "meta_info": "step2",
    }
    prompt, question = prompt_handler.get_prompt(example_question)
    print(repr(prompt))
    print(prompt_handler.format_output(question, "Sample model output text here"))

if __name__ == "__main__":
    main()