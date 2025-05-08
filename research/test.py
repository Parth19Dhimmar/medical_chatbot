import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pdf_parser.log"), logging.StreamHandler()]
)
logger = logging.getLogger("medical_pdf_parser")

# ========== Security Module ==========

class SecurityHandler:
    """Handles encryption, decryption and audit logging for sensitive medical data."""
    
    def __init__(self, key_path: str = None):
        """Initialize security handler with encryption key."""
        if key_path and os.path.exists(key_path):
            with open(key_path, "rb") as key_file:
                self.key = key_file.read()
        else:
            self.key = Fernet.generate_key()
            if key_path:
                with open(key_path, "wb") as key_file:
                    key_file.write(self.key)
        
        self.cipher = Fernet(self.key)
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def log_access(self, user_id: str, document_id: str, action: str):
        """Log access to sensitive documents for audit."""
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} - User: {user_id} - Doc: {document_id} - Action: {action}"
        logger.info(log_entry)
        
        # In production, you might want to store this in a secure database
        with open("access_audit.log", "a") as f:
            f.write(log_entry + "\n")
    
    def hash_identifier(self, identifier: str) -> str:
        """Create a one-way hash of an identifier for anonymization."""
        return hashlib.sha256(identifier.encode()).hexdigest()

# ========== Medical Dictionary Module ==========

@dataclass
class MedicalTerm:
    standard_term: str
    synonyms: List[str]
    category: str
    country_specific_variants: Dict[str, List[str]]

class MedicalDictionary:
    """Manages medical terminology across different countries and languages."""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        self.terms: Dict[str, MedicalTerm] = {}
        if dictionary_path and os.path.exists(dictionary_path):
            self.load_dictionary(dictionary_path)
        else:
            self._initialize_basic_dictionary()
    
    def _initialize_basic_dictionary(self):
        """Initialize with some basic medical terms across regions."""
        # Just a minimal example - in production this would be much more comprehensive
        self.add_term(
            "blood_pressure", 
            "Blood Pressure", 
            ["BP", "B/P", "arterial pressure"],
            "vital_signs",
            {
                "US": ["blood pressure"],
                "UK": ["blood pressure"],
                "IN": ["BP", "रक्तचाप", "रक्त दाब"],
                "ES": ["presión arterial", "presión sanguínea"]
            }
        )
        self.add_term(
            "heart_rate", 
            "Heart Rate", 
            ["HR", "pulse", "pulse rate"],
            "vital_signs",
            {
                "US": ["heart rate", "pulse"],
                "UK": ["heart rate", "pulse"],
                "IN": ["HR", "हृदय गति", "नब्ज़"],
                "ES": ["frecuencia cardíaca", "pulso"]
            }
        )
        # Many more terms would be added in a real implementation
    
    def add_term(self, term_id: str, standard_term: str, synonyms: List[str], 
                 category: str, country_variants: Dict[str, List[str]]):
        """Add a medical term to the dictionary."""
        self.terms[term_id] = MedicalTerm(
            standard_term=standard_term,
            synonyms=synonyms,
            category=category,
            country_specific_variants=country_variants
        )
    
    def get_term_variants(self, term_id: str) -> List[str]:
        """Get all variants of a term for pattern matching."""
        if term_id not in self.terms:
            return []
        
        term = self.terms[term_id]
        variants = [term.standard_term] + term.synonyms
        
        # Add country-specific variants
        for country_variants in term.country_specific_variants.values():
            variants.extend(country_variants)
        
        return variants
    
    def find_matching_term(self, text: str) -> Optional[str]:
        """Find standard term ID that matches the given text."""
        text = text.lower()
        for term_id, term_info in self.terms.items():
            # Check standard term and synonyms
            if term_info.standard_term.lower() == text or any(syn.lower() == text for syn in term_info.synonyms):
                return term_id
            
            # Check country-specific variants
            for variants in term_info.country_specific_variants.values():
                if any(variant.lower() == text for variant in variants):
                    return term_id
        
        return None
    
    def load_dictionary(self, path: str):
        """Load medical dictionary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for term_id, term_data in data.items():
            self.terms[term_id] = MedicalTerm(
                standard_term=term_data["standard_term"],
                synonyms=term_data["synonyms"],
                category=term_data["category"],
                country_specific_variants=term_data["country_specific_variants"]
            )
    
    def save_dictionary(self, path: str):
        """Save medical dictionary to JSON file."""
        data = {}
        for term_id, term_info in self.terms.items():
            data[term_id] = {
                "standard_term": term_info.standard_term,
                "synonyms": term_info.synonyms,
                "category": term_info.category,
                "country_specific_variants": term_info.country_specific_variants
            }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ========== PDF Processing Module ==========

class LayoutAnalyzer:
    """Analyzes PDF layout to identify form structure."""
    
    def analyze_layout(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze PDF structure and return layout information."""
        layout_info = {
            "form_type": None,
            "sections": [],
            "tables": [],
            "key_value_pairs": []
        }
        
        # Detect if it's a form
        if self._has_form_fields(doc):
            layout_info["form_type"] = "interactive_form"
            layout_info["form_fields"] = self._extract_form_fields(doc)
        else:
            layout_info["form_type"] = "static_document"
            
        # Extract sections, tables, and key-value pairs
        for page_num, page in enumerate(doc):
            # Find sections using headlines or formatting
            sections = self._identify_sections(page)
            for section in sections:
                section["page"] = page_num
                layout_info["sections"].append(section)
            
            # Find tables
            tables = self._identify_tables(page)
            for table in tables:
                table["page"] = page_num
                layout_info["tables"].append(table)
            
            # Find key-value pairs (common in medical forms)
            kv_pairs = self._identify_key_value_pairs(page)
            for kv in kv_pairs:
                kv["page"] = page_num
                layout_info["key_value_pairs"].append(kv)
        
        return layout_info
    
    def _has_form_fields(self, doc: fitz.Document) -> bool:
        """Check if document has interactive form fields."""
        for page in doc:
            if page.widgets():
                return True
        return False
    
    def _extract_form_fields(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract interactive form fields."""
        fields = []
        for page_num, page in enumerate(doc):
            for widget in page.widgets():
                field = {
                    "name": widget.field_name if widget.field_name else "unnamed_field",
                    "type": widget.field_type_string,
                    "value": widget.field_value,
                    "rect": widget.rect,
                    "page": page_num
                }
                fields.append(field)
        return fields
    
    def _identify_sections(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Identify document sections based on formatting and content."""
        sections = []
        text_blocks = page.get_text("dict")["blocks"]
        
        current_section = None
        
        for block in text_blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"])
                    font_size = max(span["size"] for span in line["spans"])
                    
                    # Heuristic: larger font size likely indicates a section header
                    if font_size > 12:  # Threshold can be adjusted
                        if current_section:
                            sections.append(current_section)
                        
                        current_section = {
                            "title": text,
                            "rect": line["bbox"],
                            "content_rect": [line["bbox"][0], line["bbox"][3], page.rect.width, 0]  # Will update bottom later
                        }
                    elif current_section:
                        # Update the bottom of content area
                        current_section["content_rect"][3] = max(current_section["content_rect"][3], line["bbox"][3])
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _identify_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Identify tables in the document based on layout patterns."""
        # This is a simplified placeholder - real implementation would use
        # more sophisticated algorithms for table detection
        tables = []
        
        # Get all rectangles (often used for table cells)
        paths = page.get_drawings()
        rects = []
        
        for path in paths:
            for item in path["items"]:
                if item[0] == "re":  # Rectangle
                    rects.append(item[1])  # Rectangle coordinates
        
        # Group nearby rectangles - this is a very basic approach
        # A real implementation would need clustering algorithms to identify table structures
        if len(rects) > 5:  # Arbitrary threshold
            tables.append({
                "rect": fitz.Rect(
                    min(r[0] for r in rects),
                    min(r[1] for r in rects),
                    max(r[0] + r[2] for r in rects),
                    max(r[1] + r[3] for r in rects)
                ),
                "cells": len(rects)
            })
        
        return tables
    
    def _identify_key_value_pairs(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Identify key-value pairs common in forms."""
        kv_pairs = []
        text = page.get_text("dict")
        
        # This is a simplified approach - a real implementation would be more sophisticated
        for block in text["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"])
                    
                    # Look for common key-value separator patterns
                    match = re.search(r"([^:]+):\s*(.+)", line_text)
                    if match:
                        kv_pairs.append({
                            "key": match.group(1).strip(),
                            "value": match.group(2).strip(),
                            "rect": line["bbox"]
                        })
                    else:
                        # Look for field-value layout (label followed by underline or box)
                        # This would need more sophisticated layout analysis in a real implementation
                        pass
        
        return kv_pairs

class PDFExtractor:
    """Extract medical data from PDFs with region-aware processing."""
    
    def __init__(self, medical_dict: MedicalDictionary, security: SecurityHandler):
        self.medical_dict = medical_dict
        self.security = security
        self.layout_analyzer = LayoutAnalyzer()
    
    def extract_medical_data(self, pdf_path: str, country_code: str = None) -> Dict[str, Any]:
        """Extract medical data from PDF with region-awareness."""
        # Log access for audit
        document_id = os.path.basename(pdf_path)
        user_id = "system"  # In a real system, this would be the authenticated user
        self.security.log_access(user_id, document_id, "data_extraction")
        
        # Open and analyze PDF
        doc = fitz.open(pdf_path)
        layout = self.layout_analyzer.analyze_layout(doc)
        
        # Extract medical data based on detected layout
        extracted_data = {
            "metadata": {
                "extraction_time": datetime.now().isoformat(),
                "document_id": self.security.hash_identifier(document_id),
                "country_code": country_code
            },
            "patient_info": {},
            "medical_data": {},
            "measurements": {},
            "medications": []
        }
        
        # Process based on document type
        if layout["form_type"] == "interactive_form":
            self._process_interactive_form(layout["form_fields"], extracted_data, country_code)
        else:
            self._process_static_document(doc, layout, extracted_data, country_code)
        
        # Detect and anonymize PHI data
        extracted_data = self._anonymize_phi(extracted_data)
        
        return extracted_data
    
    def _process_interactive_form(self, form_fields: List[Dict], extracted_data: Dict, country_code: str):
        """Process interactive form fields."""
        for field in form_fields:
            field_name = field["name"].lower()
            field_value = field["value"]
            
            # Skip empty fields
            if not field_value:
                continue
                
            # Map form field names to standard terms
            # This is a simplified mapping - real implementation would be more comprehensive
            if "name" in field_name:
                extracted_data["patient_info"]["name"] = field_value
            elif "birth" in field_name or "dob" in field_name:
                extracted_data["patient_info"]["date_of_birth"] = field_value
            elif "blood" in field_name and "pressure" in field_name:
                extracted_data["measurements"]["blood_pressure"] = field_value
            elif "diagnosis" in field_name:
                extracted_data["medical_data"]["diagnosis"] = field_value
            # Add more field mappings as needed
    
    def _process_static_document(self, doc: fitz.Document, layout: Dict, extracted_data: Dict, country_code: str):
        """Process static document (no interactive form fields)."""
        # Process key-value pairs first
        for kv in layout["key_value_pairs"]:
            self._process_key_value(kv, extracted_data, country_code)
        
        # Process sections
        for section in layout["sections"]:
            section_text = self._get_text_in_rect(doc[section["page"]], section["content_rect"])
            section_title = section["title"].lower()
            
            if any(term in section_title for term in ["patient", "personal"]):
                self._extract_patient_info(section_text, extracted_data)
            elif any(term in section_title for term in ["medication", "prescription", "drug"]):
                medications = self._extract_medications(section_text)
                extracted_data["medications"].extend(medications)
            elif any(term in section_title for term in ["vital", "measurement"]):
                measurements = self._extract_measurements(section_text, country_code)
                extracted_data["measurements"].update(measurements)
            elif any(term in section_title for term in ["diagnosis", "assessment"]):
                extracted_data["medical_data"]["diagnosis"] = section_text.strip()
            # Add more section mappings as needed
        
        # Process tables
        for table in layout["tables"]:
            # Simplified table processing - real implementation would be more sophisticated
            table_text = self._get_text_in_rect(doc[table["page"]], table["rect"])
            
            # Try to determine table type by looking for key terms
            if any(term in table_text.lower() for term in ["medication", "drug", "prescription"]):
                meds = self._parse_medication_table(table_text)
                extracted_data["medications"].extend(meds)
            elif any(term in table_text.lower() for term in ["lab", "test", "result"]):
                lab_results = self._parse_lab_results_table(table_text)
                if "lab_results" not in extracted_data["medical_data"]:
                    extracted_data["medical_data"]["lab_results"] = []
                extracted_data["medical_data"]["lab_results"].extend(lab_results)
    
    def _get_text_in_rect(self, page: fitz.Page, rect: List[float]) -> str:
        """Get text contained within the specified rectangle."""
        return page.get_text("text", clip=fitz.Rect(rect))
    
    def _process_key_value(self, kv: Dict, extracted_data: Dict, country_code: str):
        """Process a key-value pair."""
        key = kv["key"].lower()
        value = kv["value"]
        
        # Try to map the key to a standard term
        term_id = self.medical_dict.find_matching_term(key)
        
        if term_id:
            if term_id in ["blood_pressure", "heart_rate", "temperature", "weight", "height"]:
                if "measurements" not in extracted_data:
                    extracted_data["measurements"] = {}
                extracted_data["measurements"][term_id] = value
            # Add more term mappings as needed
        else:
            # Handle common patient fields
            if any(name_term in key for name_term in ["name", "patient name", "full name"]):
                extracted_data["patient_info"]["name"] = value
            elif any(id_term in key for id_term in ["id", "patient id", "medical record"]):
                extracted_data["patient_info"]["id"] = value
            elif any(dob_term in key for dob_term in ["birth", "dob", "born"]):
                extracted_data["patient_info"]["date_of_birth"] = value
            # Add more fields as needed
    
    def _extract_patient_info(self, text: str, extracted_data: Dict):
        """Extract patient information from text."""
        # Look for common patterns in patient info sections
        # Name pattern
        name_match = re.search(r"(?:name|patient)[\s:]+([^\n]+)", text, re.IGNORECASE)
        if name_match:
            extracted_data["patient_info"]["name"] = name_match.group(1).strip()
        
        # DOB pattern
        dob_match = re.search(r"(?:birth|dob|born)[\s:]+([^\n]+)", text, re.IGNORECASE)
        if dob_match:
            extracted_data["patient_info"]["date_of_birth"] = dob_match.group(1).strip()
        
        # Patient ID pattern
        id_match = re.search(r"(?:id|medical record|mrn)[\s:]+([^\n]+)", text, re.IGNORECASE)
        if id_match:
            extracted_data["patient_info"]["id"] = id_match.group(1).strip()
    
    def _extract_medications(self, text: str) -> List[Dict]:
        """Extract medication information from text."""
        medications = []
        
        # Look for medication patterns
        # This is a simplified approach - a real implementation would be more comprehensive
        med_matches = re.finditer(r"([A-Za-z0-9\s]+)\s+(\d+\s*(?:mg|g|ml|mcg))\s+(?:(\d+)\s+times daily|(\d+)\s+daily)", text)
        
        for match in med_matches:
            med = {
                "name": match.group(1).strip(),
                "dosage": match.group(2).strip(),
                "frequency": match.group(3) if match.group(3) else match.group(4)
            }
            medications.append(med)
        
        return medications
    
    def _extract_measurements(self, text: str, country_code: str) -> Dict[str, Any]:
        """Extract medical measurements from text."""
        measurements = {}
        
        # Blood pressure pattern
        bp_match = re.search(r"(?:blood pressure|bp|b/p)[:\s]*(\d+\s*[/\\]\s*\d+)", text, re.IGNORECASE)
        if bp_match:
            measurements["blood_pressure"] = bp_match.group(1).strip()
        
        # Heart rate pattern
        hr_match = re.search(r"(?:heart rate|pulse)[:\s]*(\d+)", text, re.IGNORECASE)
        if hr_match:
            measurements["heart_rate"] = hr_match.group(1).strip()
        
        # Temperature pattern - with regional awareness
        temp_pattern = r"(?:temperature|temp)[:\s]*(\d+\.?\d*)\s*"
        if country_code in ["US", "LR", "MM"]:  # Countries using Fahrenheit
            temp_pattern += r"°?F"
        else:  # Most countries use Celsius
            temp_pattern += r"°?C"
        
        temp_match = re.search(temp_pattern, text, re.IGNORECASE)
        if temp_match:
            measurements["temperature"] = temp_match.group(1).strip()
            # Could add unit normalization here
        
        return measurements
    
    def _parse_medication_table(self, text: str) -> List[Dict]:
        """Parse a medication table from text."""
        # This is a simplified approach - real implementation would need
        # more sophisticated table structure recognition
        medications = []
        lines = text.strip().split('\n')
        
        if len(lines) > 1:
            # Try to find header row
            header = lines[0].lower()
            if any(term in header for term in ["medication", "drug"]):
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        med = {
                            "name": parts[0],
                            "dosage": parts[1],
                            "frequency": " ".join(parts[2:])
                        }
                        medications.append(med)
        
        return medications
    
    def _parse_lab_results_table(self, text: str) -> List[Dict]:
        """Parse a lab results table from text."""
        # This is a simplified approach - real implementation would need
        # more sophisticated table structure recognition
        lab_results = []
        lines = text.strip().split('\n')
        
        if len(lines) > 1:
            # Try to find header row
            header = lines[0].lower()
            if any(term in header for term in ["test", "parameter", "lab"]):
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        result = {
                            "test": parts[0],
                            "value": parts[1],
                            "unit": parts[2] if len(parts) > 2 else "",
                            "reference_range": " ".join(parts[3:]) if len(parts) > 3 else ""
                        }
                        lab_results.append(result)
        
        return lab_results
    
    def _anonymize_phi(self, data: Dict) -> Dict:
        """Detect and anonymize Protected Health Information."""
        # Create a deep copy to avoid modifying the original
        anonymized = json.loads(json.dumps(data))
        
        # Anonymize patient identifiers
        if "patient_info" in anonymized:
            patient_info = anonymized["patient_info"]
            
            if "name" in patient_info:
                # Hash the name but keep a consistent anonymized identifier
                original_name = patient_info["name"]
                patient_info["name"] = f"PATIENT_{self.security.hash_identifier(original_name)[:8]}"
            
            if "id" in patient_info:
                # Hash the ID but keep it recognizable as an identifier
                original_id = patient_info["id"]
                patient_info["id"] = f"ID_{self.security.hash_identifier(original_id)[:8]}"
                
            # Date of birth could be changed to age or age range for less identifiable info
            if "date_of_birth" in patient_info:
                # This is simplified - a real implementation would calculate age
                patient_info["age_range"] = "REDACTED"
                del patient_info["date_of_birth"]
        
        return anonymized

# ========== Main Application ==========

class MedicalPDFParser:
    """Main application class for medical PDF parsing."""
    
    def __init__(self, medical_dict_path: str = None, encryption_key_path: str = None):
        """Initialize the medical PDF parser."""
        # Initialize security handler
        self.security = SecurityHandler(encryption_key_path)
        
        # Initialize medical dictionary
        self.medical_dict = MedicalDictionary(medical_dict_path)
        
        # Initialize PDF extractor
        self.extractor = PDFExtractor(self.medical_dict, self.security)
    
    def parse_pdf(self, pdf_path: str, country_code: str = None) -> Dict[str, Any]:
        """Parse a medical PDF and return structured data."""
        # Validate inputs
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Infer country code if not provided
        if not country_code:
            country_code = self._infer_country_code(pdf_path)
        
        # Extract data
        extracted_data = self.extractor.extract_medical_data(pdf_path, country_code)
        
        # Convert to standardized JSON format
        result = self._to_standardized_json(extracted_data)
        
        return result
    
    def _infer_country_code(self, pdf_path: str) -> str:
        """Attempt to infer country code from PDF content."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Look for country-specific patterns
        # This is a simplified approach - real implementation would be more sophisticated
        if re.search(r"\b(?:NHS|National Health Service)\b", text, re.IGNORECASE):
            return "UK"
        elif re.search(r"\b(?:Medicare|Medicaid|HIPAA)\b", text, re.IGNORECASE):
            return "US"
        elif re.search(r"\b(?:AADHAAR|CGHS|Ayushman)\b", text, re.IGNORECASE):
            return "IN"
        else:
            # Default to US if can't determine
            return "US"
    
    def _to_standardized_json(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extracted data to standardized JSON format."""
        # This would implement any final transformations needed for the output format
        # For this example, we'll just return the extracted data as is
        return extracted_data
    
    def batch_process(self, pdf_dir: str, output_dir: str):
        """Process all PDFs in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                try:
                    result = self.parse_pdf(pdf_path)
                    
                    # Save result to JSON file
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"Successfully processed {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

# ========== Example Usage ==========

def main():
    """Example usage of the medical PDF parser."""
    # Initialize parser
    parser = MedicalPDFParser(
        medical_dict_path="medical_dictionary.json",
        encryption_key_path="encryption.key"
    )
    
    # Parse a single PDF
    result = parser.parse_pdf("example_medical_record.pdf", country_code="US")
    print(json.dumps(result, indent=2))
    
    # Or batch process a directory
    # parser.batch_process("pdf_directory", "output_directory")

if __name__ == "__main__":
    main()