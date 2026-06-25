# EduVision: Voice-First Intelligent Textbook Summarizer
# Optimized for Visually Impaired Users with Always-On Voice Commands

from flask import Flask, render_template, request, jsonify, send_file
import os
import io
import base64
import json
import threading
import time
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Core libraries
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pyttsx3
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# PDF processing libraries
import PyPDF2
import pdfplumber
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# Translation and multilingual support
from googletrans import Translator
from langdetect import detect
import re

# Additional utilities
from werkzeug.utils import secure_filename
import requests
from gtts import gTTS
import tempfile
import queue
from threading import Event, Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size for PDFs

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('audio_output', exist_ok=True)
os.makedirs('pdf_pages', exist_ok=True)

# Global variables for voice control
voice_command_queue = queue.Queue()
is_always_listening = True  # Always listening for voice activation
voice_commands_active = False
current_session = None
current_language = 'en'
shutdown_event = Event()
voice_lock = Lock()

# Thread executor for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Language configurations
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'code': 'en',
        'voice_code': 'en-US',
        'tts_lang': 'en',
        'tesseract_lang': 'eng'
    },
    'hi': {
        'name': 'हिंदी (Hindi)',
        'code': 'hi',
        'voice_code': 'hi-IN',
        'tts_lang': 'hi',
        'tesseract_lang': 'hin'
    },
    'es': {
        'name': 'Español',
        'code': 'es',
        'voice_code': 'es-ES',
        'tts_lang': 'es',
        'tesseract_lang': 'spa'
    },
    'fr': {
        'name': 'Français',
        'code': 'fr',
        'voice_code': 'fr-FR',
        'tts_lang': 'fr',
        'tesseract_lang': 'fra'
    },
    'de': {
        'name': 'Deutsch',
        'code': 'de',
        'voice_code': 'de-DE',
        'tts_lang': 'de',
        'tesseract_lang': 'deu'
    }
}

# Activation phrases for voice commands
ACTIVATION_PHRASES = {
    'en': ['voice commands', 'activate voice', 'start voice', 'enable voice'],
    'hi': ['आवाज़ कमांड', 'आवाज़ चालू', 'आवाज़ शुरू', 'voice commands']
}

# Deactivation phrases
DEACTIVATION_PHRASES = {
    'en': ['stop voice', 'disable voice', 'voice off', 'stop listening'],
    'hi': ['आवाज़ बंद', 'आवाज़ रोको', 'voice off', 'stop listening']
}

# Hindi voice commands with better recognition
HINDI_COMMANDS = {
    'सारांश': 'summarize',
    'सारांश करो': 'summarize',
    'सारांश दो': 'summarize',
    'पढ़ें': 'read',
    'पढ़िए': 'read',
    'पढ़ कर सुनाएं': 'read',
    'अपलोड': 'upload',
    'अपलोड करें': 'upload',
    'मदद': 'help',
    'सहायता': 'help',
    'मदद करें': 'help',
    'दोहराएं': 'repeat',
    'दोबारा': 'repeat',
    'फिर से': 'repeat',
    'बंद': 'stop',
    'रुकें': 'pause',
    'जारी': 'continue',
    'अनुवाद': 'translate',
    'अनुवाद करें': 'translate'
}

# UI text translations
UI_TRANSLATIONS = {
    'en': {
        'title': 'EduVision - Voice-First Textbook Summarizer',
        'subtitle': 'Say "Voice Commands" to activate voice control',
        'upload_title': 'Upload Your Educational Content',
        'upload_subtitle': 'Drag and drop a file or say "Upload File"',
        'choose_file': 'Choose File',
        'pdf_docs': 'PDF Documents',
        'images': 'Images',
        'voice_active': 'Voice Commands Active',
        'voice_inactive': 'Voice Commands Inactive',
        'speak_summary': 'Speak Summary',
        'short_summary': 'Short Summary',
        'medium_summary': 'Medium Summary',
        'long_summary': 'Long Summary',
        'extracted_text': 'Extracted Text',
        'summary': 'Summary',
        'audio_summary': 'Audio Summary',
        'processing': 'Processing your content...',
        'language': 'Language',
        'translate': 'Translate'
    },
    'hi': {
        'title': 'एडुविज़न - आवाज़-प्रथम पाठ्यपुस्तक सारांशकर्ता',
        'subtitle': 'आवाज़ नियंत्रण के लिए "आवाज़ कमांड" कहें',
        'upload_title': 'अपनी शैक्षिक सामग्री अपलोड करें',
        'upload_subtitle': 'फ़ाइल खींचें या "फाइल अपलोड" कहें',
        'choose_file': 'फ़ाइल चुनें',
        'pdf_docs': 'पीडीएफ दस्तावेज़',
        'images': 'चित्र',
        'voice_active': 'आवाज़ कमांड सक्रिय',
        'voice_inactive': 'आवाज़ कमांड निष्क्रिय',
        'speak_summary': 'सारांश बोलें',
        'short_summary': 'छोटा सारांश',
        'medium_summary': 'मध्यम सारांश',
        'long_summary': 'लंबा सारांश',
        'extracted_text': 'निकाला गया पाठ',
        'summary': 'सारांश',
        'audio_summary': 'ऑडियो सारांश',
        'processing': 'आपकी सामग्री को प्रोसेस कर रहे हैं...',
        'language': 'भाषा',
        'translate': 'अनुवाद करें'
    }
}

class EduVisionCore:
    """Optimized core functionality for textbook processing"""
    
    def __init__(self):
        self.setup_models()
        self.setup_tts()
        self.setup_translator()
        
    def setup_models(self):
        """Initialize AI models with optimization"""
        try:
            # Use a smaller, faster model for better performance
            model_name = "facebook/bart-large-cnn"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                max_length=400,
                min_length=50,
                do_sample=False,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.summarizer = None
    
    def setup_tts(self):
        """Initialize optimized text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Faster speech rate for quicker feedback
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
            logger.info("TTS engine initialized")
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            self.tts_engine = None
    
    def setup_translator(self):
        """Initialize translation service with optimization"""
        try:
            self.translator = Translator()
            logger.info("Translation service initialized")
        except Exception as e:
            logger.error(f"Error initializing translator: {e}")
            self.translator = None
    
    def detect_language(self, text):
        """Fast language detection with caching"""
        try:
            if len(text.strip()) < 10:
                return 'en'
            # Use only first 500 characters for speed
            sample_text = text[:500]
            detected = detect(sample_text)
            return detected if detected in SUPPORTED_LANGUAGES else 'en'
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'
    
    def translate_text_fast(self, text, target_language='hi', source_language='auto'):
        """Optimized translation with chunking and parallel processing"""
        try:
            if not self.translator or len(text.strip()) == 0:
                return text
            
            # For very long texts, use smart truncation
            if len(text) > 8000:
                # Take first 4000 and last 4000 characters
                text = text[:4000] + "... " + text[-4000:]
            
            # Translate in smaller chunks for better performance
            chunks = self.split_text_for_translation(text, max_length=2000)
            
            if len(chunks) == 1:
                result = self.translator.translate(chunks[0], src=source_language, dest=target_language)
                return result.text
            
            # Parallel translation for multiple chunks
            translated_chunks = []
            for chunk in chunks:
                if len(chunk.strip()) > 0:
                    result = self.translator.translate(chunk, src=source_language, dest=target_language)
                    translated_chunks.append(result.text)
                else:
                    translated_chunks.append(chunk)
            
            return ' '.join(translated_chunks)
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def split_text_for_translation(self, text, max_length=2000):
        """Optimized text splitting"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_text_from_pdf_fast(self, pdf_path, language='eng', max_pages=5):
        """Optimized PDF text extraction with page limit"""
        extracted_text = ""
        
        try:
            # Method 1: Fast pdfplumber extraction with page limit
            logger.info(f"Fast PDF extraction with max {max_pages} pages...")
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = min(len(pdf.pages), max_pages)
                for page_num in range(pages_to_process):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        extracted_text += f"{text}\n"
                        # Stop early if we have enough text
                        if len(extracted_text) > 5000:
                            break
            
            if extracted_text.strip():
                logger.info(f"Fast extraction successful: {len(extracted_text)} characters")
                return extracted_text.strip()
            
        except Exception as e:
            logger.warning(f"Fast PDF extraction failed: {e}")
        
        # Fallback to PyMuPDF for difficult PDFs
        try:
            logger.info("Trying PyMuPDF fallback...")
            pdf_document = fitz.open(pdf_path)
            pages_to_process = min(pdf_document.page_count, max_pages)
            
            for page_num in range(pages_to_process):
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    extracted_text += f"{text}\n"
                    if len(extracted_text) > 5000:
                        break
            
            pdf_document.close()
            
            if extracted_text.strip():
                logger.info(f"PyMuPDF extraction successful: {len(extracted_text)} characters")
                return extracted_text.strip()
                
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        
        return ""
    
    def extract_text_from_image_fast(self, image_path, language='eng'):
        """Optimized image text extraction"""
        try:
            # Quick image preprocessing
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Resize large images for faster processing
            height, width = image.shape[:2]
            if width > 2000:
                scale = 2000 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Simple preprocessing for speed
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Fast OCR config
            config = f'--psm 6 -l {language} --oem 3'
            text = pytesseract.image_to_string(thresh, config=config)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Fast OCR failed: {e}")
            return ""
    
    def process_file_fast(self, file_path, file_type, language='eng'):
        """Fast file processing with smart limits"""
        if file_type.lower() == 'pdf':
            return self.extract_text_from_pdf_fast(file_path, language, max_pages=5)
        else:
            return self.extract_text_from_image_fast(file_path, language)
    
    def summarize_text_fast(self, text, summary_type='medium', target_language='en'):
        """Optimized text summarization"""
        if not text or len(text) < 50:
            return text
        
        try:
            # Faster length configs
            length_configs = {
                'short': {'max_length': 150, 'min_length': 30},
                'medium': {'max_length': 250, 'min_length': 60},
                'long': {'max_length': 350, 'min_length': 100}
            }
            
            config = length_configs.get(summary_type, length_configs['medium'])
            
            # Truncate very long texts for faster processing
            if len(text) > 3000:
                text = text[:3000] + "..."
            
            # Detect source language quickly
            source_lang = self.detect_language(text[:200])  # Use smaller sample
            
            # Fast summarization
            if self.summarizer and len(text) > 100:
                try:
                    summary_result = self.summarizer(
                        text,
                        max_length=config['max_length'],
                        min_length=config['min_length'],
                        do_sample=False
                    )
                    summary = summary_result[0]['summary_text']
                except Exception as e:
                    logger.warning(f"AI summarization failed: {e}")
                    summary = self.extractive_summary_fast(text, config['max_length'])
            else:
                summary = self.extractive_summary_fast(text, config['max_length'])
            
            # Translate if needed
            if target_language != 'en' and summary:
                summary = self.translate_text_fast(summary, target_language, 'en')
            
            return summary
                
        except Exception as e:
            logger.error(f"Fast summarization error: {e}")
            return self.extractive_summary_fast(text, 200)
    
    def extractive_summary_fast(self, text, max_length):
        """Fast extractive summarization"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 3:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Quick sentence selection
        important_sentences = []
        important_sentences.append(sentences[0])  # First sentence
        if len(sentences) > 2:
            important_sentences.append(sentences[len(sentences)//2])  # Middle
        if len(sentences) > 1:
            important_sentences.append(sentences[-1])  # Last
        
        summary = '. '.join(important_sentences) + '.'
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def text_to_speech_fast(self, text, language='en', filename=None):
        """Fast text-to-speech generation"""
        try:
            if not filename:
                filename = f"audio_output/speech_{int(time.time())}.mp3"
            
            # Limit text length for faster TTS
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            # Use gTTS for multilingual support
            lang_code = SUPPORTED_LANGUAGES.get(language, {}).get('tts_lang', 'en')
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(filename)
            return filename
                
        except Exception as e:
            logger.error(f"Fast TTS error: {e}")
            return None
    
    def speak_text_immediate(self, text, language='en'):
        """Immediate speech without file creation"""
        try:
            if self.tts_engine and len(text) < 200:  # Use pyttsx3 for short texts
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Create temp audio for longer texts
                temp_file = f"audio_output/temp_{int(time.time())}.mp3"
                audio_file = self.text_to_speech_fast(text, language, temp_file)
                # Note: In a real app, you'd play this audio file
        except Exception as e:
            logger.error(f"Immediate speech error: {e}")

# Initialize core functionality
eduvision = EduVisionCore()

class VoiceCommandHandler:
    """Always-on voice command handler optimized for visually impaired users"""
    
    def __init__(self, core):
        self.core = core
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Optimize recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
        # Adjust for ambient noise
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Voice recognition optimized for always-on listening")
        except Exception as e:
            logger.error(f"Error setting up voice recognition: {e}")
        
        self.commands = {
            'summarize': self.handle_summarize,
            'read': self.handle_read,
            'upload': self.handle_upload,
            'help': self.handle_help,
            'repeat': self.handle_repeat,
            'stop': self.handle_stop,
            'pause': self.handle_pause,
            'continue': self.handle_continue,
            'translate': self.handle_translate,
            'short': self.handle_short_summary,
            'medium': self.handle_medium_summary,
            'long': self.handle_long_summary
        }
        self.last_response = ""
        self.is_paused = False
    
    def always_listen_for_activation(self):
        """Always listen for voice activation phrases"""
        global voice_commands_active, is_always_listening
        
        logger.info("Starting always-on voice activation listening...")
        
        while is_always_listening and not shutdown_event.is_set():
            try:
                with voice_lock:
                    with self.microphone as source:
                        # Short timeout for activation phrases
                        audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=3)
                
                # Try multiple languages for activation
                activation_detected = False
                spoken_text = ""
                
                for lang_code in ['en-US', 'hi-IN']:
                    try:
                        spoken_text = self.recognizer.recognize_google(audio, language=lang_code).lower()
                        logger.info(f"Heard: '{spoken_text}' in {lang_code}")
                        
                        # Check for activation phrases
                        current_lang = 'hi' if lang_code == 'hi-IN' else 'en'
                        activation_phrases = ACTIVATION_PHRASES.get(current_lang, [])
                        
                        for phrase in activation_phrases:
                            if phrase in spoken_text:
                                activation_detected = True
                                logger.info(f"Voice activation detected: '{phrase}'")
                                break
                        
                        if activation_detected:
                            break
                            
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition error: {e}")
                        continue
                
                if activation_detected:
                    if not voice_commands_active:
                        voice_commands_active = True
                        logger.info("Voice commands activated!")
                        
                        # Provide audio feedback
                        if 'hindi' in spoken_text or 'हिंदी' in spoken_text:
                            feedback = "आवाज़ कमांड चालू हो गए। आप अब बोल सकते हैं।"
                            global current_language
                            current_language = 'hi'
                        else:
                            feedback = "Voice commands activated. You can now speak your commands."
                        
                        self.core.speak_text_immediate(feedback, current_language)
                        
                        # Start command processing thread
                        command_thread = threading.Thread(
                            target=self.listen_for_commands, 
                            args=(current_language,)
                        )
                        command_thread.daemon = True
                        command_thread.start()
                
                elif voice_commands_active:
                    # Check for deactivation phrases
                    deactivation_phrases = DEACTIVATION_PHRASES.get(current_language, [])
                    for phrase in deactivation_phrases:
                        if phrase in spoken_text:
                            voice_commands_active = False
                            feedback = "आवाज़ कमांड बंद हो गए।" if current_language == 'hi' else "Voice commands deactivated."
                            self.core.speak_text_immediate(feedback, current_language)
                            logger.info("Voice commands deactivated")
                            break
                            
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in always-on listening: {e}")
                time.sleep(1)
    
    def listen_for_commands(self, language='en'):
        """Listen for actual voice commands when activated"""
        global voice_commands_active
        
        logger.info(f"Listening for voice commands in {language}")
        
        while voice_commands_active and not shutdown_event.is_set():
            try:
                with voice_lock:
                    with self.microphone as source:
                        # Longer timeout for commands
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Recognize speech in the selected language
                lang_code = SUPPORTED_LANGUAGES.get(language, {}).get('voice_code', 'en-US')
                command = self.recognizer.recognize_google(audio, language=lang_code).lower()
                logger.info(f"Voice command received: '{command}' in {language}")
                
                # Quick acknowledgment
                if language == 'hi':
                    self.core.speak_text_immediate("समझ गया", language)
                else:
                    self.core.speak_text_immediate("Got it", language)
                
                # Process command
                self.process_command(command, language)
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                logger.debug("Could not understand audio")
                continue
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in command listening: {e}")
                time.sleep(1)
    
    def process_command(self, command, language='en'):
        """Process voice command with better matching"""
        command = command.strip().lower()
        
        # Handle Hindi commands
        if language == 'hi':
            for hindi_cmd, english_cmd in HINDI_COMMANDS.items():
                if hindi_cmd in command or any(word in command for word in hindi_cmd.split()):
                    command = english_cmd
                    logger.info(f"Hindi command mapped: {hindi_cmd} -> {english_cmd}")
                    break
        
        # Find and execute matching command
        for cmd_key, handler in self.commands.items():
            if cmd_key in command or any(word in command for word in cmd_key.split()):
                logger.info(f"Executing command: {cmd_key}")
                handler(command, language)
                return
        
        # Handle general queries
        self.handle_general_query(command, language)
    
    def handle_summarize(self, command, language='en'):
        """Handle summarize commands with immediate feedback"""
        if current_session and current_session.get('summary'):
            response = current_session['summary']
            self.core.speak_text_immediate(response, language)
            self.last_response = response
        else:
            if language == 'hi':
                response = "पहले कोई फाइल अपलोड करें।"
            else:
                response = "Please upload a file first."
            self.core.speak_text_immediate(response, language)
            self.last_response = response
    
    def handle_read(self, command, language='en'):
        """Handle read commands"""
        if self.last_response:
            self.core.speak_text_immediate(self.last_response, language)
        else:
            if language == 'hi':
                response = "पढ़ने के लिए कुछ नहीं है।"
            else:
                response = "Nothing to read."
            self.core.speak_text_immediate(response, language)
    
    def handle_upload(self, command, language='en'):
        """Handle upload commands"""
        if language == 'hi':
            response = "कृपया वेबसाइट पर फाइल अपलोड करें।"
        else:
            response = "Please upload a file using the website interface."
        self.core.speak_text_immediate(response, language)
        self.last_response = response
    
    def handle_help(self, command, language='en'):
        """Handle help commands with concise instructions"""
        if language == 'hi':
            help_text = """मैं आपकी मदद कर सकता हूं। आप कह सकते हैं: सारांश, पढ़ें, छोटा सारांश, लंबा सारांश, अनुवाद, या आवाज़ बंद।"""
        else:
            help_text = """I can help you. You can say: summarize, read, short summary, long summary, translate, or stop voice."""
        
        self.core.speak_text_immediate(help_text, language)
        self.last_response = help_text
    
    def handle_short_summary(self, command, language='en'):
        """Handle short summary request"""
        if current_session and current_session.get('original_text'):
            summary = eduvision.summarize_text_fast(
                current_session['original_text'], 
                'short', 
                language
            )
            current_session['summary'] = summary
            self.core.speak_text_immediate(summary, language)
            self.last_response = summary
        else:
            self.handle_upload(command, language)
    
    def handle_medium_summary(self, command, language='en'):
        """Handle medium summary request"""
        if current_session and current_session.get('original_text'):
            summary = eduvision.summarize_text_fast(
                current_session['original_text'], 
                'medium', 
                language
            )
            current_session['summary'] = summary
            self.core.speak_text_immediate(summary, language)
            self.last_response = summary
        else:
            self.handle_upload(command, language)
    
    def handle_long_summary(self, command, language='en'):
        """Handle long summary request"""
        if current_session and current_session.get('original_text'):
            summary = eduvision.summarize_text_fast(
                current_session['original_text'], 
                'long', 
                language
            )
            current_session['summary'] = summary
            self.core.speak_text_immediate(summary, language)
            self.last_response = summary
        else:
            self.handle_upload(command, language)
    
    def handle_translate(self, command, language='en'):
        """Handle translation requests"""
        if current_session and current_session.get('original_text'):
            # Toggle between English and Hindi
            target_lang = 'hi' if language == 'en' else 'en'
            translated = eduvision.translate_text_fast(
                current_session['summary'] or current_session['original_text'], 
                target_lang
            )
            self.core.speak_text_immediate(translated, target_lang)
            self.last_response = translated
        else:
            self.handle_upload(command, language)
    
    def handle_repeat(self, command, language='en'):
        """Handle repeat commands"""
        if self.last_response:
            self.core.speak_text_immediate(self.last_response, language)
        else:
            if language == 'hi':
                response = "दोहराने के लिए कुछ नहीं है।"
            else:
                response = "Nothing to repeat."
            self.core.speak_text_immediate(response, language)
    
    def handle_stop(self, command, language='en'):
        """Handle stop commands"""
        global voice_commands_active
        voice_commands_active = False
        if language == 'hi':
            response = "आवाज़ कमांड बंद हो गए।"
        else:
            response = "Voice commands stopped."
        self.core.speak_text_immediate(response, language)
    
    def handle_pause(self, command, language='en'):
        """Handle pause commands"""
        self.is_paused = True
        if language == 'hi':
            response = "रुक गए।"
        else:
            response = "Paused."
        self.core.speak_text_immediate(response, language)
    
    def handle_continue(self, command, language='en'):
        """Handle continue commands"""
        self.is_paused = False
        if language == 'hi':
            response = "जारी।"
        else:
            response = "Continuing."
        self.core.speak_text_immediate(response, language)
    
    def handle_general_query(self, query, language='en'):
        """Handle general queries with helpful responses"""
        if language == 'hi':
            response = f"मैंने सुना: {query}। मदद के लिए 'मदद' कहें।"
        else:
            response = f"I heard: {query}. Say 'help' for available commands."
        
        self.core.speak_text_immediate(response, language)
        self.last_response = response

# Initialize voice command handler
voice_handler = VoiceCommandHandler(eduvision)

# Start always-on voice activation listener
def start_voice_activation():
    """Start the always-on voice activation listener"""
    activation_thread = threading.Thread(target=voice_handler.always_listen_for_activation)
    activation_thread.daemon = True
    activation_thread.start()
    logger.info("Always-on voice activation started")

# Flask Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/get_languages')
def get_languages():
    """Get supported languages"""
    return jsonify({'languages': SUPPORTED_LANGUAGES})

@app.route('/get_translations/<language>')
def get_translations(language):
    """Get UI translations for a language"""
    translations = UI_TRANSLATIONS.get(language, UI_TRANSLATIONS['en'])
    return jsonify({'translations': translations})

@app.route('/get_voice_status')
def get_voice_status():
    """Get current voice command status"""
    return jsonify({
        'always_listening': is_always_listening,
        'commands_active': voice_commands_active,
        'current_language': current_language
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Optimized file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get language settings
        target_language = request.form.get('target_language', 'en')
        ocr_language = request.form.get('ocr_language', 'eng')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Immediate voice feedback
            if voice_commands_active:
                if target_language == 'hi':
                    eduvision.speak_text_immediate("फाइल मिल गई, प्रोसेसिंग शुरू कर रहे हैं।", target_language)
                else:
                    eduvision.speak_text_immediate("File received, starting processing.", target_language)
            
            # Determine file type
            file_extension = filename.rsplit('.', 1)[1].lower()
            file_type = 'pdf' if file_extension == 'pdf' else 'image'
            
            # Fast processing with language support
            tesseract_lang = SUPPORTED_LANGUAGES.get(target_language, {}).get('tesseract_lang', 'eng')
            if ocr_language != 'auto':
                tesseract_lang = ocr_language
            
            # Use optimized processing
            extracted_text = eduvision.process_file_fast(filepath, file_type, tesseract_lang)
            
            if not extracted_text:
                error_msg = f'Could not extract text from {file_type}'
                if voice_commands_active:
                    if target_language == 'hi':
                        eduvision.speak_text_immediate("फाइल से टेक्स्ट नहीं निकाला जा सका।", target_language)
                    else:
                        eduvision.speak_text_immediate("Could not extract text from file.", target_language)
                return jsonify({'error': error_msg}), 400
            
            # Detect source language
            detected_language = eduvision.detect_language(extracted_text)
            
            # Clean text
            cleaned_text = extracted_text.strip()
            
            # Fast summarization
            summary_type = request.form.get('summary_type', 'medium')
            summary = eduvision.summarize_text_fast(cleaned_text, summary_type, target_language)
            
            # Fast translation if needed
            translated_text = cleaned_text
            if target_language != 'en' and detected_language != target_language:
                translated_text = eduvision.translate_text_fast(cleaned_text, target_language, detected_language)
            
            # Fast audio generation
            audio_file = eduvision.text_to_speech_fast(summary, target_language)
            
            # Update global session
            global current_session, current_language
            current_session = {
                'original_text': cleaned_text,
                'translated_text': translated_text,
                'summary': summary,
                'audio_file': audio_file,
                'file_type': file_type,
                'filename': filename,
                'detected_language': detected_language,
                'target_language': target_language
            }
            current_language = target_language
            voice_handler.last_response = summary
            
            # Voice feedback for completion
            if voice_commands_active:
                if target_language == 'hi':
                    eduvision.speak_text_immediate("प्रोसेसिंग पूरी हुई। सारांश के लिए 'सारांश' कहें।", target_language)
                else:
                    eduvision.speak_text_immediate("Processing complete. Say 'summarize' to hear the summary.", target_language)
            
            # Return truncated text for faster response
            return jsonify({
                'success': True,
                'original_text': cleaned_text[:1500] + "..." if len(cleaned_text) > 1500 else cleaned_text,
                'translated_text': translated_text[:1500] + "..." if len(translated_text) > 1500 else translated_text,
                'summary': summary,
                'audio_file': audio_file,
                'word_count': len(cleaned_text.split()),
                'summary_word_count': len(summary.split()),
                'file_type': file_type,
                'filename': filename,
                'detected_language': detected_language,
                'target_language': target_language
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        if voice_commands_active:
            eduvision.speak_text_immediate("Processing failed. Please try again.", current_language)
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/voice_command', methods=['POST'])
def voice_command():
    """Handle voice command requests (mainly for web interface)"""
    try:
        data = request.json
        command = data.get('command', '').lower()
        language = data.get('language', current_language)
        
        if command == 'get_status':
            return jsonify({
                'always_listening': is_always_listening,
                'commands_active': voice_commands_active,
                'current_language': current_language
            })
        
        elif command == 'speak_summary':
            if current_session and current_session.get('summary'):
                # Create audio file for web playback
                audio_file = eduvision.text_to_speech_fast(
                    current_session['summary'], 
                    current_session.get('target_language', 'en')
                )
                return jsonify({
                    'success': True, 
                    'message': 'Audio generated',
                    'audio_file': audio_file
                })
            else:
                return jsonify({'error': 'No summary available'}), 400
        
        else:
            # Process command through voice handler
            voice_handler.process_command(command, language)
            return jsonify({'success': True, 'message': 'Command processed'})
            
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        return jsonify({'error': 'Voice command failed'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    try:
        return send_file(f'audio_output/{filename}')
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return "Audio file not found", 404

@app.route('/get_summary', methods=['POST'])
def get_summary():
    """Get different types of summaries with fast processing"""
    try:
        data = request.json
        summary_type = data.get('type', 'medium')
        target_language = data.get('target_language', current_language)
        
        if current_session and current_session.get('original_text'):
            # Fast summarization
            summary = eduvision.summarize_text_fast(
                current_session['original_text'], 
                summary_type,
                target_language
            )
            
            # Fast audio generation
            audio_file = eduvision.text_to_speech_fast(summary, target_language)
            
            # Update session
            current_session['summary'] = summary
            current_session['audio_file'] = audio_file
            current_session['target_language'] = target_language
            voice_handler.last_response = summary
            
            return jsonify({
                'success': True,
                'summary': summary,
                'audio_file': audio_file,
                'word_count': len(summary.split()),
                'target_language': target_language
            })
        
        return jsonify({'error': 'No content available for summarization'}), 400
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return jsonify({'error': 'Summary generation failed'}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """Fast translation endpoint"""
    try:
        data = request.json
        text = data.get('text', '')
        target_language = data.get('target_language', 'hi')
        source_language = data.get('source_language', 'auto')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Use fast translation
        translated = eduvision.translate_text_fast(text, target_language, source_language)
        
        return jsonify({
            'success': True,
            'translated_text': translated,
            'source_language': source_language,
            'target_language': target_language
        })
        
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return jsonify({'error': 'Translation failed'}), 500

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'pdf'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cleanup function
def cleanup_on_shutdown():
    """Clean up resources on shutdown"""
    global is_always_listening, voice_commands_active
    is_always_listening = False
    voice_commands_active = False
    shutdown_event.set()
    logger.info("Voice services shut down cleanly")

import atexit
atexit.register(cleanup_on_shutdown)

if __name__ == '__main__':
    # Create directories
    os.makedirs('templates', exist_ok=True)
    
    # Start voice activation immediately
    start_voice_activation()
    
    # Provide initial voice feedback
    time.sleep(1)  # Wait for initialization
    welcome_msg_en = "EduVision is ready. Say 'Voice Commands' to start voice control."
    welcome_msg_hi = "एडुविज़न तैयार है। आवाज़ नियंत्रण के लिए 'आवाज़ कमांड' कहें।"
    
    try:
        eduvision.speak_text_immediate(welcome_msg_en, 'en')
        time.sleep(1)
        eduvision.speak_text_immediate(welcome_msg_hi, 'hi')
    except:
        logger.info("Voice feedback not available, but system is ready")
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)