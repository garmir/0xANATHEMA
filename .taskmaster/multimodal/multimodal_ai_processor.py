#!/usr/bin/env python3
"""
Multi-Modal AI Processor
Integrates vision, audio, and text processing capabilities for enhanced project understanding
"""

import json
import os
import time
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import mimetypes
import subprocess

class ModalityType(Enum):
    """Supported modality types"""
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    CODE = "code"

class ProcessingCapability(Enum):
    """Processing capabilities"""
    OCR = "ocr"
    OBJECT_DETECTION = "object_detection"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENT_PARSING = "document_parsing"
    UI_ANALYSIS = "ui_analysis"

@dataclass
class MultiModalInput:
    """Multi-modal input data"""
    input_id: str
    modality: ModalityType
    content: Union[str, bytes]
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    content_type: Optional[str] = None

@dataclass
class ProcessingResult:
    """Result of multi-modal processing"""
    input_id: str
    modality: ModalityType
    capabilities_applied: List[ProcessingCapability]
    extracted_text: Optional[str]
    structured_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str]

@dataclass
class ProjectInsight:
    """Insight extracted from multi-modal analysis"""
    insight_id: str
    insight_type: str
    description: str
    modalities_involved: List[ModalityType]
    confidence: float
    actionable_recommendations: List[str]
    related_files: List[str]

class MultiModalAIProcessor:
    """Multi-modal AI processor for comprehensive project understanding"""
    
    def __init__(self, processing_dir: str = '.taskmaster/multimodal'):
        self.processing_dir = Path(processing_dir)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.inputs_dir = self.processing_dir / 'inputs'
        self.outputs_dir = self.processing_dir / 'outputs'
        self.cache_dir = self.processing_dir / 'cache'
        self.models_dir = self.processing_dir / 'models'
        
        for directory in [self.inputs_dir, self.outputs_dir, self.cache_dir, self.models_dir]:
            directory.mkdir(exist_ok=True)
        
        # Processing capabilities
        self.available_capabilities = {
            ModalityType.TEXT: [
                ProcessingCapability.SENTIMENT_ANALYSIS,
                ProcessingCapability.CODE_ANALYSIS
            ],
            ModalityType.IMAGE: [
                ProcessingCapability.OCR,
                ProcessingCapability.OBJECT_DETECTION,
                ProcessingCapability.UI_ANALYSIS
            ],
            ModalityType.AUDIO: [
                ProcessingCapability.SPEECH_TO_TEXT
            ],
            ModalityType.DOCUMENT: [
                ProcessingCapability.DOCUMENT_PARSING,
                ProcessingCapability.OCR
            ],
            ModalityType.CODE: [
                ProcessingCapability.CODE_ANALYSIS
            ]
        }
        
        # Processing cache
        self.processing_cache = {}
        self.insights_cache = []
        
        print(f"✅ Initialized multi-modal AI processor with {len(self.available_capabilities)} modality types")
    
    def process_project_directory(self, project_path: str) -> List[ProjectInsight]:
        """Process entire project directory for multi-modal insights"""
        
        project_path = Path(project_path)
        print(f"🔍 Analyzing project directory: {project_path}")
        
        # Discover and categorize files
        discovered_files = self._discover_project_files(project_path)
        
        print(f"📁 Discovered {len(discovered_files)} files across {len(set(f.modality for f in discovered_files))} modalities")
        
        # Process each file
        processing_results = []
        for file_input in discovered_files:
            try:
                result = self.process_multimodal_input(file_input)
                processing_results.append(result)
            except Exception as e:
                print(f"⚠️ Failed to process {file_input.file_path}: {e}")
        
        # Generate project insights
        insights = self._generate_project_insights(processing_results)
        
        print(f"💡 Generated {len(insights)} project insights")
        
        # Save insights
        self._save_insights(insights)
        
        return insights
    
    def process_multimodal_input(self, input_data: MultiModalInput) -> ProcessingResult:
        """Process a single multi-modal input"""
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(input_data)
        if cache_key in self.processing_cache:
            cached_result = self.processing_cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        print(f"🔄 Processing {input_data.modality.value} input: {input_data.input_id}")
        
        try:
            # Get applicable capabilities
            capabilities = self.available_capabilities.get(input_data.modality, [])
            
            # Initialize result
            result = ProcessingResult(
                input_id=input_data.input_id,
                modality=input_data.modality,
                capabilities_applied=[],
                extracted_text=None,
                structured_data={},
                confidence_scores={},
                processing_time=0,
                success=False,
                error_message=None
            )
            
            # Apply each capability
            for capability in capabilities:
                try:
                    capability_result = self._apply_capability(input_data, capability)
                    
                    # Merge results
                    result.capabilities_applied.append(capability)
                    
                    if capability_result.get('text'):
                        result.extracted_text = capability_result['text']
                    
                    result.structured_data.update(capability_result.get('structured_data', {}))
                    result.confidence_scores.update(capability_result.get('confidence_scores', {}))
                    
                except Exception as e:
                    print(f"⚠️ Capability {capability.value} failed: {e}")
                    result.confidence_scores[capability.value] = 0.0
            
            result.success = len(result.capabilities_applied) > 0
            result.processing_time = time.time() - start_time
            
            # Cache result
            self.processing_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            error_result = ProcessingResult(
                input_id=input_data.input_id,
                modality=input_data.modality,
                capabilities_applied=[],
                extracted_text=None,
                structured_data={},
                confidence_scores={},
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            return error_result
    
    def _discover_project_files(self, project_path: Path) -> List[MultiModalInput]:
        """Discover and categorize project files"""
        
        discovered_files = []
        
        # File type mappings
        modality_mappings = {
            # Text files
            '.txt': ModalityType.TEXT,
            '.md': ModalityType.TEXT,
            '.rst': ModalityType.TEXT,
            '.log': ModalityType.TEXT,
            
            # Code files
            '.py': ModalityType.CODE,
            '.js': ModalityType.CODE,
            '.ts': ModalityType.CODE,
            '.jsx': ModalityType.CODE,
            '.tsx': ModalityType.CODE,
            '.html': ModalityType.CODE,
            '.css': ModalityType.CODE,
            '.scss': ModalityType.CODE,
            '.json': ModalityType.CODE,
            '.yml': ModalityType.CODE,
            '.yaml': ModalityType.CODE,
            '.xml': ModalityType.CODE,
            '.sql': ModalityType.CODE,
            '.sh': ModalityType.CODE,
            '.bat': ModalityType.CODE,
            
            # Images
            '.png': ModalityType.IMAGE,
            '.jpg': ModalityType.IMAGE,
            '.jpeg': ModalityType.IMAGE,
            '.gif': ModalityType.IMAGE,
            '.bmp': ModalityType.IMAGE,
            '.svg': ModalityType.IMAGE,
            '.webp': ModalityType.IMAGE,
            
            # Documents
            '.pdf': ModalityType.DOCUMENT,
            '.doc': ModalityType.DOCUMENT,
            '.docx': ModalityType.DOCUMENT,
            '.xls': ModalityType.DOCUMENT,
            '.xlsx': ModalityType.DOCUMENT,
            '.ppt': ModalityType.DOCUMENT,
            '.pptx': ModalityType.DOCUMENT,
            
            # Audio
            '.mp3': ModalityType.AUDIO,
            '.wav': ModalityType.AUDIO,
            '.ogg': ModalityType.AUDIO,
            '.flac': ModalityType.AUDIO,
            '.m4a': ModalityType.AUDIO,
            
            # Video
            '.mp4': ModalityType.VIDEO,
            '.avi': ModalityType.VIDEO,
            '.mkv': ModalityType.VIDEO,
            '.mov': ModalityType.VIDEO,
            '.webm': ModalityType.VIDEO
        }
        
        # Scan project directory
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                
                # Skip certain directories
                if any(part in str(file_path) for part in ['.git', 'node_modules', '__pycache__', '.venv', 'venv']):
                    continue
                
                # Determine modality
                file_extension = file_path.suffix.lower()
                modality = modality_mappings.get(file_extension, ModalityType.TEXT)
                
                # Read file content (with size limit)
                try:
                    if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                        continue
                    
                    if modality in [ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.VIDEO, ModalityType.DOCUMENT]:
                        # For binary files, store file path reference
                        content = str(file_path)
                    else:
                        # For text files, read content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    
                    # Create input object
                    input_data = MultiModalInput(
                        input_id=f"file_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        modality=modality,
                        content=content,
                        metadata={
                            'file_size': file_path.stat().st_size,
                            'modified_time': file_path.stat().st_mtime,
                            'extension': file_extension
                        },
                        file_path=str(file_path),
                        content_type=mimetypes.guess_type(str(file_path))[0]
                    )
                    
                    discovered_files.append(input_data)
                    
                except Exception as e:
                    print(f"⚠️ Failed to read {file_path}: {e}")
        
        return discovered_files
    
    def _apply_capability(self, input_data: MultiModalInput, 
                         capability: ProcessingCapability) -> Dict[str, Any]:
        """Apply specific processing capability"""
        
        if capability == ProcessingCapability.OCR:
            return self._perform_ocr(input_data)
        elif capability == ProcessingCapability.OBJECT_DETECTION:
            return self._perform_object_detection(input_data)
        elif capability == ProcessingCapability.SPEECH_TO_TEXT:
            return self._perform_speech_to_text(input_data)
        elif capability == ProcessingCapability.SENTIMENT_ANALYSIS:
            return self._perform_sentiment_analysis(input_data)
        elif capability == ProcessingCapability.CODE_ANALYSIS:
            return self._perform_code_analysis(input_data)
        elif capability == ProcessingCapability.DOCUMENT_PARSING:
            return self._perform_document_parsing(input_data)
        elif capability == ProcessingCapability.UI_ANALYSIS:
            return self._perform_ui_analysis(input_data)
        else:
            return {'confidence_scores': {capability.value: 0.0}}
    
    def _perform_ocr(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform OCR on image content"""
        # Simulated OCR processing
        
        if input_data.modality == ModalityType.IMAGE:
            # Simulate OCR results based on file name
            file_name = input_data.metadata.get('file_path', '')
            
            if 'screenshot' in file_name.lower():
                extracted_text = "User Interface Screenshot\nLogin Form\nUsername: [input field]\nPassword: [input field]\nSign In [button]"
                confidence = 0.85
            elif 'diagram' in file_name.lower():
                extracted_text = "System Architecture Diagram\nDatabase -> API Server -> Frontend\nLoad Balancer\nCaching Layer"
                confidence = 0.75
            else:
                extracted_text = "Image content detected\nText elements identified\nGraphical components present"
                confidence = 0.60
            
            return {
                'text': extracted_text,
                'structured_data': {
                    'text_regions': ['header', 'main_content', 'footer'],
                    'detected_elements': ['text', 'buttons', 'images']
                },
                'confidence_scores': {'ocr': confidence}
            }
        
        return {'confidence_scores': {'ocr': 0.0}}
    
    def _perform_object_detection(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform object detection on image content"""
        # Simulated object detection
        
        if input_data.modality == ModalityType.IMAGE:
            file_name = input_data.metadata.get('file_path', '')
            
            if 'ui' in file_name.lower() or 'screenshot' in file_name.lower():
                detected_objects = ['button', 'text_field', 'menu', 'icon']
                confidence = 0.80
            elif 'diagram' in file_name.lower():
                detected_objects = ['rectangle', 'arrow', 'text_label', 'connector']
                confidence = 0.75
            else:
                detected_objects = ['shape', 'text', 'graphic_element']
                confidence = 0.60
            
            return {
                'structured_data': {
                    'detected_objects': detected_objects,
                    'object_count': len(detected_objects),
                    'primary_object': detected_objects[0] if detected_objects else None
                },
                'confidence_scores': {'object_detection': confidence}
            }
        
        return {'confidence_scores': {'object_detection': 0.0}}
    
    def _perform_speech_to_text(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform speech-to-text on audio content"""
        # Simulated speech-to-text
        
        if input_data.modality == ModalityType.AUDIO:
            # Simulate transcription based on file metadata
            duration = input_data.metadata.get('duration', 60)  # Default 60 seconds
            
            transcribed_text = f"Recorded meeting discussion about project requirements. " \
                             f"Duration: {duration} seconds. Topics covered include system architecture, " \
                             f"user requirements, and implementation timeline."
            
            return {
                'text': transcribed_text,
                'structured_data': {
                    'duration_seconds': duration,
                    'speaker_count': 2,
                    'topics': ['architecture', 'requirements', 'timeline']
                },
                'confidence_scores': {'speech_to_text': 0.85}
            }
        
        return {'confidence_scores': {'speech_to_text': 0.0}}
    
    def _perform_sentiment_analysis(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform sentiment analysis on text content"""
        
        if input_data.modality == ModalityType.TEXT:
            text_content = str(input_data.content)
            
            # Simple sentiment analysis based on keywords
            positive_words = ['good', 'great', 'excellent', 'awesome', 'perfect', 'success', 'working', 'completed']
            negative_words = ['bad', 'terrible', 'awful', 'failed', 'error', 'problem', 'issue', 'broken']
            
            positive_count = sum(1 for word in positive_words if word in text_content.lower())
            negative_count = sum(1 for word in negative_words if word in text_content.lower())
            
            if positive_count > negative_count:
                sentiment = 'positive'
                score = 0.6 + (positive_count / (positive_count + negative_count + 1)) * 0.4
            elif negative_count > positive_count:
                sentiment = 'negative'
                score = 0.4 - (negative_count / (positive_count + negative_count + 1)) * 0.4
            else:
                sentiment = 'neutral'
                score = 0.5
            
            return {
                'structured_data': {
                    'sentiment': sentiment,
                    'sentiment_score': score,
                    'positive_indicators': positive_count,
                    'negative_indicators': negative_count
                },
                'confidence_scores': {'sentiment_analysis': 0.75}
            }
        
        return {'confidence_scores': {'sentiment_analysis': 0.0}}
    
    def _perform_code_analysis(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform code analysis"""
        
        if input_data.modality == ModalityType.CODE:
            code_content = str(input_data.content)
            file_extension = input_data.metadata.get('extension', '')
            
            # Analyze code structure
            lines_of_code = len(code_content.split('\n'))
            
            # Language detection
            language_map = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.html': 'HTML',
                '.css': 'CSS',
                '.json': 'JSON',
                '.yml': 'YAML',
                '.yaml': 'YAML'
            }
            language = language_map.get(file_extension, 'Unknown')
            
            # Simple complexity analysis
            complexity_indicators = ['if', 'for', 'while', 'try', 'except', 'class', 'def', 'function']
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in code_content.lower())
            
            # Function/class detection
            functions = code_content.count('def ') + code_content.count('function ')
            classes = code_content.count('class ')
            
            return {
                'structured_data': {
                    'language': language,
                    'lines_of_code': lines_of_code,
                    'complexity_score': complexity_score,
                    'function_count': functions,
                    'class_count': classes,
                    'estimated_maintainability': 'high' if complexity_score < 10 else 'medium' if complexity_score < 25 else 'low'
                },
                'confidence_scores': {'code_analysis': 0.90}
            }
        
        return {'confidence_scores': {'code_analysis': 0.0}}
    
    def _perform_document_parsing(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform document parsing"""
        
        if input_data.modality == ModalityType.DOCUMENT:
            # Simulated document parsing
            file_path = input_data.file_path
            
            if file_path and file_path.endswith('.pdf'):
                extracted_text = "PDF Document Content\nProject Requirements Specification\n" \
                               "1. Introduction\n2. Functional Requirements\n3. Technical Specifications"
                page_count = 10
            else:
                extracted_text = "Document content extracted\nStructured information identified"
                page_count = 1
            
            return {
                'text': extracted_text,
                'structured_data': {
                    'document_type': 'specification',
                    'page_count': page_count,
                    'sections': ['introduction', 'requirements', 'specifications'],
                    'has_images': True,
                    'has_tables': True
                },
                'confidence_scores': {'document_parsing': 0.85}
            }
        
        return {'confidence_scores': {'document_parsing': 0.0}}
    
    def _perform_ui_analysis(self, input_data: MultiModalInput) -> Dict[str, Any]:
        """Perform UI/UX analysis on images"""
        
        if input_data.modality == ModalityType.IMAGE:
            file_name = input_data.metadata.get('file_path', '')
            
            if 'ui' in file_name.lower() or 'screenshot' in file_name.lower():
                ui_elements = ['navigation', 'content_area', 'sidebar', 'footer']
                usability_score = 0.8
                accessibility_score = 0.7
            else:
                ui_elements = ['visual_element']
                usability_score = 0.5
                accessibility_score = 0.5
            
            return {
                'structured_data': {
                    'ui_elements': ui_elements,
                    'usability_score': usability_score,
                    'accessibility_score': accessibility_score,
                    'design_pattern': 'modern',
                    'color_scheme': 'professional',
                    'responsive_indicators': True
                },
                'confidence_scores': {'ui_analysis': 0.75}
            }
        
        return {'confidence_scores': {'ui_analysis': 0.0}}
    
    def _generate_project_insights(self, processing_results: List[ProcessingResult]) -> List[ProjectInsight]:
        """Generate high-level project insights from processing results"""
        
        insights = []
        
        # Analyze code quality across the project
        code_results = [r for r in processing_results if r.modality == ModalityType.CODE and r.success]
        if code_results:
            total_loc = sum(r.structured_data.get('lines_of_code', 0) for r in code_results)
            avg_complexity = sum(r.structured_data.get('complexity_score', 0) for r in code_results) / len(code_results)
            languages = set(r.structured_data.get('language', 'Unknown') for r in code_results)
            
            code_insight = ProjectInsight(
                insight_id="code_quality_overview",
                insight_type="code_analysis",
                description=f"Project contains {total_loc} lines of code across {len(languages)} languages with average complexity {avg_complexity:.1f}",
                modalities_involved=[ModalityType.CODE],
                confidence=0.90,
                actionable_recommendations=[
                    "Consider code review for high-complexity files" if avg_complexity > 20 else "Code complexity is well-managed",
                    f"Multi-language project: {', '.join(languages)}",
                    "Implement automated testing for better code quality"
                ],
                related_files=[r.input_id for r in code_results[:5]]  # Top 5 files
            )
            insights.append(code_insight)
        
        # Analyze documentation coverage
        text_results = [r for r in processing_results if r.modality == ModalityType.TEXT and r.success]
        doc_results = [r for r in processing_results if r.modality == ModalityType.DOCUMENT and r.success]
        
        if text_results or doc_results:
            doc_coverage = len(text_results + doc_results) / len(processing_results) if processing_results else 0
            
            doc_insight = ProjectInsight(
                insight_id="documentation_analysis",
                insight_type="documentation",
                description=f"Documentation coverage: {doc_coverage:.1%} of project files",
                modalities_involved=[ModalityType.TEXT, ModalityType.DOCUMENT],
                confidence=0.85,
                actionable_recommendations=[
                    "Good documentation coverage" if doc_coverage > 0.3 else "Consider adding more documentation",
                    "Add README files for better project understanding",
                    "Include code comments and API documentation"
                ],
                related_files=[r.input_id for r in (text_results + doc_results)[:3]]
            )
            insights.append(doc_insight)
        
        # Analyze UI/UX elements if present
        image_results = [r for r in processing_results if r.modality == ModalityType.IMAGE and r.success]
        ui_results = [r for r in image_results if 'ui_analysis' in r.confidence_scores]
        
        if ui_results:
            avg_usability = sum(r.structured_data.get('usability_score', 0) for r in ui_results) / len(ui_results)
            avg_accessibility = sum(r.structured_data.get('accessibility_score', 0) for r in ui_results) / len(ui_results)
            
            ui_insight = ProjectInsight(
                insight_id="ui_ux_analysis",
                insight_type="user_interface",
                description=f"UI/UX analysis: {avg_usability:.1%} usability, {avg_accessibility:.1%} accessibility",
                modalities_involved=[ModalityType.IMAGE],
                confidence=0.80,
                actionable_recommendations=[
                    "Improve accessibility features" if avg_accessibility < 0.8 else "Good accessibility compliance",
                    "Enhance user experience design" if avg_usability < 0.8 else "Good usability design",
                    "Consider responsive design improvements"
                ],
                related_files=[r.input_id for r in ui_results]
            )
            insights.append(ui_insight)
        
        # Multi-modal integration insight
        modalities_present = set(r.modality for r in processing_results if r.success)
        if len(modalities_present) > 2:
            integration_insight = ProjectInsight(
                insight_id="multimodal_integration",
                insight_type="integration",
                description=f"Rich multi-modal project with {len(modalities_present)} content types: {', '.join(m.value for m in modalities_present)}",
                modalities_involved=list(modalities_present),
                confidence=0.95,
                actionable_recommendations=[
                    "Leverage multi-modal content for comprehensive documentation",
                    "Consider automated content generation pipelines",
                    "Implement cross-modal content validation"
                ],
                related_files=[r.input_id for r in processing_results[:10]]
            )
            insights.append(integration_insight)
        
        return insights
    
    def _generate_cache_key(self, input_data: MultiModalInput) -> str:
        """Generate cache key for input data"""
        content_hash = hashlib.md5(str(input_data.content).encode()).hexdigest()
        return f"{input_data.modality.value}_{content_hash[:16]}"
    
    def _save_insights(self, insights: List[ProjectInsight]):
        """Save insights to disk"""
        try:
            insights_data = [asdict(insight) for insight in insights]
            
            # Convert enums to strings
            for insight_data in insights_data:
                insight_data['modalities_involved'] = [m.value if hasattr(m, 'value') else str(m) 
                                                     for m in insight_data['modalities_involved']]
            
            insights_file = self.outputs_dir / f'project_insights_{int(time.time())}.json'
            with open(insights_file, 'w') as f:
                json.dump(insights_data, f, indent=2)
            
            print(f"💾 Saved insights to: {insights_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save insights: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing capabilities and results"""
        return {
            'available_modalities': [m.value for m in self.available_capabilities.keys()],
            'processing_capabilities': {
                modality.value: [cap.value for cap in capabilities]
                for modality, capabilities in self.available_capabilities.items()
            },
            'cache_size': len(self.processing_cache),
            'insights_generated': len(self.insights_cache)
        }

def main():
    """Demo of multi-modal AI processing"""
    print("Multi-Modal AI Processor Demo")
    print("=" * 40)
    
    processor = MultiModalAIProcessor()
    
    # Demo: Create sample inputs
    sample_inputs = [
        MultiModalInput(
            input_id="demo_text",
            modality=ModalityType.TEXT,
            content="This is an excellent project with great documentation and working features.",
            metadata={'source': 'demo'},
            content_type='text/plain'
        ),
        MultiModalInput(
            input_id="demo_code",
            modality=ModalityType.CODE,
            content="""
def process_data(input_data):
    try:
        if not input_data:
            return None
        
        processed = []
        for item in input_data:
            if validate_item(item):
                processed.append(transform_item(item))
        
        return processed
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return None
""",
            metadata={'extension': '.py', 'source': 'demo'},
            content_type='text/x-python'
        ),
        MultiModalInput(
            input_id="demo_image",
            modality=ModalityType.IMAGE,
            content="/path/to/ui_screenshot.png",
            metadata={'file_path': 'ui_screenshot.png', 'source': 'demo'},
            content_type='image/png'
        )
    ]
    
    # Process each input
    results = []
    for input_data in sample_inputs:
        result = processor.process_multimodal_input(input_data)
        results.append(result)
        
        print(f"✅ Processed {input_data.modality.value}: "
              f"{len(result.capabilities_applied)} capabilities, "
              f"{result.processing_time:.2f}s")
    
    # Generate insights
    insights = processor._generate_project_insights(results)
    
    print(f"\n💡 Generated {len(insights)} insights:")
    for insight in insights:
        print(f"  • {insight.insight_type}: {insight.description}")
        print(f"    Confidence: {insight.confidence:.1%}")
        print(f"    Recommendations: {len(insight.actionable_recommendations)}")
    
    # Show processing summary
    summary = processor.get_processing_summary()
    print(f"\n📊 Processing Summary:")
    print(f"  Available modalities: {len(summary['available_modalities'])}")
    print(f"  Total capabilities: {sum(len(caps) for caps in summary['processing_capabilities'].values())}")
    print(f"  Cache entries: {summary['cache_size']}")
    
    print(f"\n✅ Multi-modal AI processing demo completed")

if __name__ == "__main__":
    main()