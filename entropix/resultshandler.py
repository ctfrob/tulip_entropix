import json
import re
from datetime import datetime
from pathlib import Path
import logging

class ResultsHandler:
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%m_%d_%Y_%H%M")
        self.output_file = self.output_dir / f"benchmark_results_{timestamp}_eval.jsonl"
        self.full_output_file = self.output_dir / f"benchmark_results_{timestamp}_fulleval.jsonl"
        
        # Support both answer formats
        self.correct_patterns = [
            # For original format (CORRECT ANSWER: X)
            re.compile(r"CORRECT\s+ANSWER:\s*([A-H])"),
            # For new format (full text answer)
            re.compile(r"CORRECT\s+ANSWER:\s*(.*?)(?=\s*\n|\s*$)"),
            # For new format with answer_idx
            re.compile(r"CORRECT\s+ANSWER_IDX:\s*([A-H])")
        ]
        
        # Statistics
        self.total_processed = 0
        self.successful_extracts = 0
        self.failed_extracts = 0
        
        logging.info(f"Initialized ResultsHandler. Output file: {self.output_file}")

    def find_selected_answer(self, text: str) -> tuple:
        """
        Find selected answer with flexible pattern matching
        Returns: tuple(letter_answer, full_text_answer)
        """
        # Look for letter answer first
        letter_patterns = [
            r"(?:\*\*)?Selected Answer:?\s*([A-H])",
            r"(?:\*\*)?Selected\s*Answer\s*(?:is|=)?\s*([A-H])",
            r"Answer:\s*([A-H])",
            r"(?:chosen|final|selected)?\s*answer\s*(?:is|:|=)?\s*([A-H])",
            r"(?:\*\*)?Selected Answer Index:?\s*([A-H])",
            r"Answer Index:\s*([A-H])",
            r"(?:chosen|final|selected)?\s*answer_idx\s*(?:is|:|=)?\s*([A-H])"
        ]
        
        # Look for full text answer
        text_patterns = [
            r"(?:\*\*)?Selected Answer:?\s*(.*?)(?=\s*\n|\s*$)",
            r"(?:\*\*)?Final Answer:?\s*(.*?)(?=\s*\n|\s*$)",
            r"Answer:\s*(.*?)(?=\s*\n|\s*$)"
        ]
    
        letter_answer = None
        full_text_answer = None

        # Try to find letter answer
        for pattern in letter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                letter_answer = match.group(1)
                break
    
        # Try to find full text answer
        for pattern in text_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                full_text_answer = match.group(1).strip()
                break

        # Last resort: find first standalone A-H after any mention of answer
        if not letter_answer:
            if "answer" in text.lower():
                after_answer = text[text.lower().find("answer"):]
                letter_match = re.search(r'\b([A-H])\b', after_answer)
                if letter_match:
                    letter_answer = letter_match.group(1)
            
        return letter_answer, full_text_answer

    def find_correct_answer(self, text: str) -> tuple:
        """Find correct answer using multiple patterns
        Returns: tuple(letter_answer, full_text_answer)
        """
        letter_answer = None
        full_text_answer = None
        
        # Try to find letter answer first (original format or answer_idx)
        for pattern in [self.correct_patterns[0], self.correct_patterns[2]]:
            match = pattern.search(text)
            if match:
                letter_answer = match.group(1)
                break
                
        # Try to find full text answer
        match = self.correct_patterns[1].search(text)
        if match:
            full_text_answer = match.group(1).strip()
            
        return letter_answer, full_text_answer

    def process_result(self, output_text: str, question_id: str = None) -> None:
        try:
            correct_letter, correct_text = self.find_correct_answer(output_text)
            selected_letter, selected_text = self.find_selected_answer(output_text)
            
            # Determine which format to use based on what we found
            if not correct_letter and not correct_text:
                self.failed_extracts += 1
                logging.warning(f"Failed to extract answers for question {question_id}. "
                              f"Found correct letter: {correct_letter}, text: {correct_text}")
                return
            
            # Basic result for eval file
            result = {
                "question_id": question_id,
                "timestamp": datetime.now().isoformat(),
                "correct_answer_letter": correct_letter,
                "correct_answer_text": correct_text,
                "selected_answer_letter": selected_letter,
                "selected_answer_text": selected_text,
                "is_correct": (correct_letter and selected_letter and correct_letter == selected_letter) or 
                             (correct_text and selected_text and correct_text.lower() == selected_text.lower())
            }
        
            # Full result including model output
            full_result = {
                **result,
                "full_output": output_text
            }
        
            # Write both files
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
            with open(self.full_output_file, 'a') as f:
                f.write(json.dumps(full_result) + '\n')
        
            self.successful_extracts += 1
        
        except Exception as e:
            logging.error(f"Error processing result for question {question_id}: {e}")
            self.failed_extracts += 1
    
        self.total_processed += 1

    def get_stats(self) -> dict:
        """Return current processing statistics"""
        return {
            "total_processed": self.total_processed,
            "successful_extracts": self.successful_extracts,
            "failed_extracts": self.failed_extracts,
            "success_rate": f"{(self.successful_extracts/self.total_processed)*100:.2f}%" if self.total_processed > 0 else "0%"
        }

    def finalize(self) -> None:
        """Log final statistics and perform any cleanup"""
        stats = self.get_stats()
        logging.info(f"Benchmark complete. Final statistics: {json.dumps(stats, indent=2)}")
        
        # Convert JSONL to CSV if needed
        try:
            csv_file = self.output_file.with_suffix('.csv')
            with open(self.output_file, 'r') as jsonl_file, open(csv_file, 'w') as csv_out:
                # Write header with new columns
                csv_out.write("question_id,timestamp,correct_answer_letter,correct_answer_text,"
                            "selected_answer_letter,selected_answer_text,is_correct\n")
                
                # Convert each JSONL line to CSV
                for line in jsonl_file:
                    data = json.loads(line)
                    csv_out.write(f"{data.get('question_id', 'NA')},{data['timestamp']},"
                                f"{data.get('correct_answer_letter', '')},{data.get('correct_answer_text', '')},"
                                f"{data.get('selected_answer_letter', '')},{data.get('selected_answer_text', '')},"
                                f"{data['is_correct']}\n")
            
            logging.info(f"Created CSV version of results at: {csv_file}")
        except Exception as e:
            logging.error(f"Failed to create CSV version: {e}")