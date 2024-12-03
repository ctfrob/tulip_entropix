from typing import Optional, Dict, List
import time
import logging
from dataclasses import dataclass
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class ConceptExtractorConfig:
    model: str = "gpt-4-turbo-preview"  # Can fallback to gpt-3.5-turbo if needed
    temperature: float = 0.3
    max_tokens: int = 500
    concept_template: str = """You are an expert medical system supporting a medical doctor with concept extraction. You are not diagnosing the patient, but extracting relevant medical concepts from the question.
Extract ALL relevant medical concepts from the question:

Primary: List ALL diagnoses, injuries, or conditions
Risk Factors: List ALL symptoms, signs, test results, demographics, and relevant history
Assessment Areas: List ALL relevant medical specialties and anatomical areas

Be thorough and specific. Include ALL relevant findings.

Question: {question}
"""

class OpenAIConceptExtractor:
    def __init__(self, config: Optional[ConceptExtractorConfig] = None):
        self.config = config or ConceptExtractorConfig()
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def extract_concepts(self, question: str) -> str:
        """Extract medical concepts using OpenAI API"""
        start_time = time.time()
        
        try:
            prompt = self.config.concept_template.format(question=question)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a medical concept extraction system. Always respond in the exact format requested with Primary, Risk Factors, and Assessment Areas sections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            concepts = self._clean_output(response.choices[0].message.content)
            
            elapsed = time.time() - start_time
            logger.info(f"Concept extraction completed in {elapsed:.2f}s")
            logger.debug(f"Extracted concepts:\n{concepts}")
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error in concept extraction: {e}")
            return ""

    def _clean_output(self, text: str) -> str:
        """Clean and format concept extraction output"""
        try:
            # Initialize default structure
            concept_dict = {
                'Primary': [],
                'Risk Factors': [],
                'Assessment Areas': []
            }

            current_section = None
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if 'Primary:' in line:
                    current_section = 'Primary'
                    content = line.split('Primary:')[1].strip()
                elif 'Risk Factors:' in line:
                    current_section = 'Risk Factors'
                    content = line.split('Risk Factors:')[1].strip()
                elif 'Assessment Areas:' in line:
                    current_section = 'Assessment Areas'
                    content = line.split('Assessment Areas:')[1].strip()
                else:
                    content = line
                    
                if current_section and content:
                    # Split content if it contains commas
                    items = [item.strip() for item in content.split(',') if item.strip()]
                    concept_dict[current_section].extend(items)

            # Format output
            return "\n".join([
                f"Primary: {', '.join(concept_dict['Primary'])}",
                f"Risk Factors: {', '.join(concept_dict['Risk Factors'])}",
                f"Assessment Areas: {', '.join(concept_dict['Assessment Areas'])}"
            ])

        except Exception as e:
            logger.error(f"Error in clean_output: {e}")
            return ""

    def close(self):
        """Cleanup resources if needed"""
        pass  # OpenAI client doesn't need explicit cleanup