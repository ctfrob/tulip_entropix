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
        
        # Keep correct answer pattern the same
        self.correct_pattern = re.compile(r"CORRECT\s+ANSWER:\s*([A-H])")
        
        # Statistics
        self.total_processed = 0
        self.successful_extracts = 0
        self.failed_extracts = 0
        
        logging.info(f"Initialized ResultsHandler. Output file: {self.output_file}")

    def find_selected_answer(self, text: str) -> str:
        """Find selected answer with flexible pattern matching"""
        # Multiple patterns to catch different formats
        patterns = [
            r"(?:\*\*)?Selected Answer:?\s*([A-H])",  # Matches with/without asterisks, optional colon, flexible spacing
            r"(?:\*\*)?Selected\s*Answer\s*(?:is|=)?\s*([A-H])",  # Matches variants with "is" or "="
            r"Answer:\s*([A-H])",  # Simple "Answer: X" format
            r"(?:chosen|final|selected)?\s*answer\s*(?:is|:|=)?\s*([A-H])"  # Case insensitive, very flexible
        ]
    
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
    
        # Last resort: find first standalone A-H after any mention of "answer"
        if "answer" in text.lower():
            after_answer = text[text.lower().find("answer"):]
            letter_match = re.search(r'\b([A-H])\b', after_answer)
            if letter_match:
                return letter_match.group(1)
            
        return None

    def process_result(self, output_text: str, question_id: str = None) -> None:
        try:
            correct_match = self.correct_pattern.search(output_text)
            selected_answer = self.find_selected_answer(output_text)
            
            if not correct_match or not selected_answer:
                self.failed_extracts += 1
                return
            
            # Basic result for eval file
            result = {
                "question_id": question_id,
                "timestamp": datetime.now().isoformat(),
                "correct_answer": correct_match.group(1),
                "selected_answer": selected_answer,
                "is_correct": correct_match.group(1) == selected_answer
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
                # Write header
                csv_out.write("question_id,timestamp,correct_answer,selected_answer,is_correct\n")
                
                # Convert each JSONL line to CSV
                for line in jsonl_file:
                    data = json.loads(line)
                    csv_out.write(f"{data.get('question_id', 'NA')},{data['timestamp']},"
                                f"{data['correct_answer']},{data['selected_answer']},{data['is_correct']}\n")
            
            logging.info(f"Created CSV version of results at: {csv_file}")
        except Exception as e:
            logging.error(f"Failed to create CSV version: {e}")