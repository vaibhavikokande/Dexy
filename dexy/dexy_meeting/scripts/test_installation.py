#!/usr/bin/env python3
"""
Test Installation Script for Dexy Meeting Agent
This script tests all components of the meeting agent installation
"""

import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
import subprocess
import platform

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class InstallationTester:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'tests': {}
        }
        
    def get_system_info(self):
        """Get detailed system information"""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'current_dir': str(Path.cwd()),
            'project_root': str(self.project_root)
        }
    
    def print_header(self):
        """Print test header"""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 60)
        print("    🧪 DEXY MEETING AGENT INSTALLATION TEST")
        print("=" * 60)
        print(f"{Colors.END}")
        print(f"{Colors.BLUE}Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
        print(f"{Colors.BLUE}Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}{Colors.END}")
        print(f"{Colors.BLUE}Platform: {platform.system()} {platform.machine()}{Colors.END}")
        print()
    
    def test_environment_setup(self):
        """Test environment setup"""
        print(f"{Colors.YELLOW}🔧 Testing Environment Setup...{Colors.END}")
        
        tests = {}
        
        # Test directories
        required_dirs = [
            'data/audio',
            'data/transcripts',
            'data/summaries', 
            'data/memory',
            'logs'
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            tests[f'directory_{dir_path.replace("/", "_")}'] = exists
            print(f"  {'✅' if exists else '❌'} Directory: {dir_path}")
        
        # Test configuration files
        config_files = ['.env', '.env.example', '.gitignore', 'requirements.txt']
        
        for file_path in config_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            tests[f'config_{file_path.replace(".", "_")}'] = exists
            print(f"  {'✅' if exists else '❌'} Config file: {file_path}")
        
        self.test_results['tests']['environment_setup'] = tests
        
        success_count = sum(tests.values())
        total_count = len(tests)
        
        print(f"  Result: {success_count}/{total_count} checks passed")
        return success_count == total_count
    
    def test_python_dependencies(self):
        """Test Python dependencies"""
        print(f"{Colors.YELLOW}📦 Testing Python Dependencies...{Colors.END}")
        
        critical_packages = [
            'langchain',
            'openai',
            'streamlit',
            'whisper',
            'faiss',
            'dotenv',
            'pydub',
            'gtts',
            'speech_recognition',
            'pandas',
            'numpy',
            'plotly'
        ]
        
        optional_packages = [
            'sounddevice',
            'pyaudio',
            'pygame',
            'torch',
            'transformers',
            'chromadb',
            'tiktoken',
            'fastapi',
            'uvicorn'
        ]
        
        tests = {}
        
        # Test critical packages
        critical_success = 0
        for package in critical_packages:
            try:
                __import__(package)
                tests[f'critical_{package}'] = True
                critical_success += 1
                print(f"  ✅ Critical: {package}")
            except ImportError as e:
                tests[f'critical_{package}'] = False
                print(f"  ❌ Critical: {package} - {str(e)}")
        
        # Test optional packages
        optional_success = 0
        for package in optional_packages:
            try:
                __import__(package)
                tests[f'optional_{package}'] = True
                optional_success += 1
                print(f"  ✅ Optional: {package}")
            except ImportError as e:
                tests[f'optional_{package}'] = False
                print(f"  ⚠️ Optional: {package} - {str(e)}")
        
        self.test_results['tests']['python_dependencies'] = tests
        
        print(f"  Critical: {critical_success}/{len(critical_packages)} packages")
        print(f"  Optional: {optional_success}/{len(optional_packages)} packages")
        
        return critical_success == len(critical_packages)
    
    def test_system_dependencies(self):
        """Test system dependencies"""
        print(f"{Colors.YELLOW}⚙️ Testing System Dependencies...{Colors.END}")
        
        tests = {}
        
        # Test ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, text=True, timeout=5)
            ffmpeg_available = result.returncode == 0
            tests['ffmpeg'] = ffmpeg_available
            print(f"  {'✅' if ffmpeg_available else '❌'} FFmpeg")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tests['ffmpeg'] = False
            print(f"  ❌ FFmpeg - Not found or timeout")
        
        # Test audio system
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            audio_available = len(devices) > 0
            tests['audio_devices'] = audio_available
            print(f"  {'✅' if audio_available else '❌'} Audio devices ({len(devices)} found)")
        except Exception as e:
            tests['audio_devices'] = False
            print(f"  ❌ Audio devices - {str(e)}")
        
        # Test microphone
        try:
            import sounddevice as sd
            import numpy as np
            
            # Try to record a short sample
            duration = 0.1  # 100ms
            sample_rate = 44100
            
            recording = sd.rec(int(duration * sample_rate), 
                             samplerate=sample_rate, channels=1)
            sd.wait()
            
            microphone_available = recording is not None and len(recording) > 0
            tests['microphone'] = microphone_available
            print(f"  {'✅' if microphone_available else '❌'} Microphone access")
        except Exception as e:
            tests['microphone'] = False
            print(f"  ❌ Microphone access - {str(e)}")
        
        self.test_results['tests']['system_dependencies'] = tests
        
        # System dependencies are nice to have but not critical
        return True
    
    def test_core_modules(self):
        """Test core application modules"""
        print(f"{Colors.YELLOW}🧠 Testing Core Modules...{Colors.END}")
        
        tests = {}
        
        # Test core module imports
        core_modules = [
            'core.transcriber',
            'core.summarizer', 
            'core.memory_manager',
            'core.emailer',
            'core.wake_word',
            'core.tts'
        ]
        
        for module in core_modules:
            try:
                __import__(module)
                tests[f'import_{module.replace(".", "_")}'] = True
                print(f"  ✅ Import: {module}")
            except ImportError as e:
                tests[f'import_{module.replace(".", "_")}'] = False
                print(f"  ❌ Import: {module} - {str(e)}")
            except Exception as e:
                tests[f'import_{module.replace(".", "_")}'] = False
                print(f"  ⚠️ Import: {module} - {str(e)}")
        
        self.test_results['tests']['core_modules'] = tests
        
        # Don't require all modules to be importable yet
        return True
    
    def test_configuration(self):
        """Test configuration loading"""
        print(f"{Colors.YELLOW}⚙️ Testing Configuration...{Colors.END}")
        
        tests = {}
        
        # Test .env file loading
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check for required environment variables
            required_vars = [
                'OPENAI_API_KEY',
                'EMAIL_HOST',
                'EMAIL_USER',
                'WAKE_WORD'
            ]
            
            env_vars_set = 0
            for var in required_vars:
                value = os.getenv(var)
                is_set = value is not None and value != f"your_{var.lower()}_here"
                tests[f'env_var_{var.lower()}'] = is_set
                
                if is_set:
                    env_vars_set += 1
                    print(f"  ✅ Environment variable: {var}")
                else:
                    print(f"  ⚠️ Environment variable: {var} - Not set or default value")
            
            tests['env_file_loaded'] = True
            print(f"  Variables configured: {env_vars_set}/{len(required_vars)}")
            
        except Exception as e:
            tests['env_file_loaded'] = False
            print(f"  ❌ Environment file loading - {str(e)}")
        
        # Test config module
        try:
            from config import settings
            tests['config_module'] = True
            print(f"  ✅ Config module loaded")
        except Exception as e:
            tests['config_module'] = False
            print(f"  ❌ Config module - {str(e)}")
        
        self.test_results['tests']['configuration'] = tests
        
        # Configuration issues are warnings, not failures
        return True
    
    def test_openai_connection(self):
        """Test OpenAI API connection"""
        print(f"{Colors.YELLOW}🤖 Testing OpenAI Connection...{Colors.END}")
        
        tests = {}
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key or api_key == 'your_openai_api_key_here':
                tests['api_key_present'] = False
                tests['api_connection'] = False
                print(f"  ⚠️ OpenAI API key not configured")
            else:
                tests['api_key_present'] = True
                
                try:
                    import openai
                    
                    # Test a simple API call
                    client = openai.OpenAI(api_key=api_key)
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    
                    tests['api_connection'] = True
                    print(f"  ✅ OpenAI API connection successful")
                    
                except Exception as e:
                    tests['api_connection'] = False
                    print(f"  ❌ OpenAI API connection failed - {str(e)}")
                    
        except Exception as e:
            tests['api_key_present'] = False
            tests['api_connection'] = False
            print(f"  ❌ OpenAI setup failed - {str(e)}")
        
        self.test_results['tests']['openai_connection'] = tests
        
        # OpenAI connection is not required for basic functionality
        return True
    
    def test_whisper_model(self):
        """Test Whisper model loading"""
        print(f"{Colors.YELLOW}🎤 Testing Whisper Model...{Colors.END}")
        
        tests = {}
        
        try:
            import whisper
            
            # Try to load the smallest model
            print(f"  Loading Whisper model (this may take a moment)...")
            model = whisper.load_model("base")
            
            tests['whisper_model_load'] = True
            print(f"  ✅ Whisper model loaded successfully")
            
            # Test transcription with a dummy audio
            try:
                import numpy as np
                
                # Create dummy audio (1 second of silence)
                dummy_audio = np.zeros(16000, dtype=np.float32)
                result = model.transcribe(dummy_audio)
                
                tests['whisper_transcription'] = True
                print(f"  ✅ Whisper transcription test passed")
                
            except Exception as e:
                tests['whisper_transcription'] = False
                print(f"  ⚠️ Whisper transcription test failed - {str(e)}")
                
        except Exception as e:
            tests['whisper_model_load'] = False
            tests['whisper_transcription'] = False
            print(f"  ❌ Whisper model loading failed - {str(e)}")
        
        self.test_results['tests']['whisper_model'] = tests
        
        return tests.get('whisper_model_load', False)
    
    def test_faiss_vector_store(self):
        """Test FAISS vector store"""
        print(f"{Colors.YELLOW}🧠 Testing FAISS Vector Store...{Colors.END}")
        
        tests = {}
        
        try:
            import faiss
            import numpy as np
            
            # Create a simple vector store
            dimension = 384
            index = faiss.IndexFlatL2(dimension)
            
            # Add some dummy vectors
            vectors = np.random.random((10, dimension)).astype('float32')
            index.add(vectors)
            
            # Test search
            query = np.random.random((1, dimension)).astype('float32')
            distances, indices = index.search(query, 5)
            
            tests['faiss_index_creation'] = True
            tests['faiss_vector_operations'] = True
            print(f"  ✅ FAISS vector store working")
            
        except Exception as e:
            tests['faiss_index_creation'] = False
            tests['faiss_vector_operations'] = False
            print(f"  ❌ FAISS vector store failed - {str(e)}")
        
        self.test_results['tests']['faiss_vector_store'] = tests
        
        return tests.get('faiss_index_creation', False)
    
    def test_email_functionality(self):
        """Test email functionality"""
        print(f"{Colors.YELLOW}📧 Testing Email Functionality...{Colors.END}")
        
        tests = {}
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Test email configuration
            from dotenv import load_dotenv
            load_dotenv()
            
            email_host = os.getenv('EMAIL_HOST')
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')
            
            if not all([email_host, email_user, email_password]):
                tests['email_config'] = False
                tests['email_connection'] = False
                print(f"  ⚠️ Email configuration incomplete")
            else:
                tests['email_config'] = True
                print(f"  ✅ Email configuration found")
                
                # Note: We don't actually test SMTP connection to avoid spam
                # This would require user confirmation
                tests['email_connection'] = None
                print(f"  ℹ️ Email connection test skipped (requires user confirmation)")
                
        except Exception as e:
            tests['email_config'] = False
            tests['email_connection'] = False
            print(f"  ❌ Email setup failed - {str(e)}")
        
        self.test_results['tests']['email_functionality'] = tests
        
        return True
    
    def test_streamlit_app(self):
        """Test Streamlit application"""
        print(f"{Colors.YELLOW}🌐 Testing Streamlit App...{Colors.END}")
        
        tests = {}
        
        try:
            import streamlit as st
            
            # Test if streamlit app file exists
            app_file = self.project_root / 'ui' / 'streamlit_app.py'
            tests['streamlit_app_file'] = app_file.exists()
            print(f"  {'✅' if app_file.exists() else '❌'} Streamlit app file")
            
            # Test streamlit installation
            tests['streamlit_import'] = True
            print(f"  ✅ Streamlit import successful")
            
        except Exception as e:
            tests['streamlit_import'] = False
            print(f"  ❌ Streamlit test failed - {str(e)}")
        
        self.test_results['tests']['streamlit_app'] = tests
        
        return tests.get('streamlit_import', False)
    
    def test_audio_processing(self):
        """Test audio processing capabilities"""
        print(f"{Colors.YELLOW}🔊 Testing Audio Processing...{Colors.END}")
        
        tests = {}
        
        # Test pydub
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
            
            # Create a test audio segment
            tone = Sine(440).to_audio_segment(duration=100)  # 100ms, 440Hz
            
            tests['pydub_audio_generation'] = True
            print(f"  ✅ Pydub audio generation")
            
        except Exception as e:
            tests['pydub_audio_generation'] = False
            print(f"  ❌ Pydub audio generation - {str(e)}")
        
        # Test TTS
        try:
            from gtts import gTTS
            
            # Create a test TTS
            tts = gTTS(text="Hello, this is a test", lang='en')
            
            tests['tts_generation'] = True
            print(f"  ✅ TTS generation")
            
        except Exception as e:
            tests['tts_generation'] = False
            print(f"  ❌ TTS generation - {str(e)}")
        
        # Test speech recognition
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            tests['speech_recognition'] = True
            print(f"  ✅ Speech recognition")
            
        except Exception as e:
            tests['speech_recognition'] = False
            print(f"  ❌ Speech recognition - {str(e)}")
        
        self.test_results['tests']['audio_processing'] = tests
        
        return any(tests.values())
    
    def test_memory_performance(self):
        """Test memory usage and performance"""
        print(f"{Colors.YELLOW}🚀 Testing Memory Performance...{Colors.END}")
        
        tests = {}
        
        try:
            import psutil
            import time
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate some workload
            start_time = time.time()
            
            # Import heavy modules
            import numpy as np
            import pandas as pd
            
            # Create some data
            data = np.random.random((1000, 100))
            df = pd.DataFrame(data)
            
            # Simple operations
            result = df.sum().sum()
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = final_memory - initial_memory
            processing_time = end_time - start_time
            
            tests['memory_usage'] = memory_increase < 500  # Less than 500MB increase
            tests['processing_speed'] = processing_time < 10  # Less than 10 seconds
            
            print(f"  Memory increase: {memory_increase:.1f}MB")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  {'✅' if tests['memory_usage'] else '⚠️'} Memory usage acceptable")
            print(f"  {'✅' if tests['processing_speed'] else '⚠️'} Processing speed acceptable")
            
        except Exception as e:
            tests['memory_usage'] = False
            tests['processing_speed'] = False
            print(f"  ❌ Performance test failed - {str(e)}")
        
        self.test_results['tests']['memory_performance'] = tests
        
        return True
    
    def save_test_results(self):
        """Save test results to file"""
        results_file = self.project_root / 'logs' / 'test_results.json'
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"{Colors.BLUE}📊 Test results saved to {results_file}{Colors.END}")
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}📋 TEST SUMMARY{Colors.END}")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category, tests in self.test_results['tests'].items():
            category_passed = 0
            category_total = 0
            
            for test_name, result in tests.items():
                category_total += 1
                total_tests += 1
                
                if result is True:
                    category_passed += 1
                    passed_tests += 1
                elif result is False:
                    failed_tests += 1
                else:  # None or skipped
                    skipped_tests += 1
            
            status_color = Colors.GREEN if category_passed == category_total else Colors.YELLOW
            print(f"{status_color}{category.replace('_', ' ').title()}: {category_passed}/{category_total}{Colors.END}")
        
        print("=" * 50)
        print(f"{Colors.GREEN}✅ Passed: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}❌ Failed: {failed_tests}{Colors.END}")
        print(f"{Colors.YELLOW}⚠️ Skipped: {skipped_tests}{Colors.END}")
        print(f"{Colors.BLUE}📊 Total: {total_tests}{Colors.END}")
        
        # Overall status
        critical_failures = self.get_critical_failures()
        
        if critical_failures:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ CRITICAL ISSUES FOUND:{Colors.END}")
            for failure in critical_failures:
                print(f"  - {failure}")
            print(f"\n{Colors.YELLOW}Please fix these issues before using the application.{Colors.END}")
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ INSTALLATION LOOKS GOOD!{Colors.END}")
            print(f"{Colors.BLUE}You can now start using the Dexy Meeting Agent.{Colors.END}")
    
    def get_critical_failures(self):
        """Get list of critical failures"""
        critical_failures = []
        
        # Check critical dependencies
        python_deps = self.test_results['tests'].get('python_dependencies', {})
        critical_packages = ['langchain', 'openai', 'streamlit', 'whisper', 'faiss']
        
        for package in critical_packages:
            if not python_deps.get(f'critical_{package}', False):
                critical_failures.append(f"Missing critical package: {package}")
        
        # Check environment setup
        env_setup = self.test_results['tests'].get('environment_setup', {})
        if not env_setup.get('config__env', False):
            critical_failures.append("Missing .env configuration file")
        
        return critical_failures
    
    def run_all_tests(self):
        """Run all tests"""
        self.print_header()
        
        test_functions = [
            ("Environment Setup", self.test_environment_setup),
            ("Python Dependencies", self.test_python_dependencies),
            ("System Dependencies", self.test_system_dependencies),
            ("Core Modules", self.test_core_modules),
            ("Configuration", self.test_configuration),
            ("OpenAI Connection", self.test_openai_connection),
            ("Whisper Model", self.test_whisper_model),
            ("FAISS Vector Store", self.test_faiss_vector_store),
            ("Email Functionality", self.test_email_functionality),
            ("Streamlit App", self.test_streamlit_app),
            ("Audio Processing", self.test_audio_processing),
            ("Memory Performance", self.test_memory_performance)
        ]
        
        for test_name, test_func in test_functions:
            try:
                print(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
                success = test_func()
                
                if success:
                    print(f"{Colors.GREEN}✅ {test_name} completed successfully{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}⚠️ {test_name} completed with issues{Colors.END}")
                    
            except Exception as e:
                print(f"{Colors.RED}❌ {test_name} failed with error: {str(e)}{Colors.END}")
                print(f"{Colors.RED}Traceback: {traceback.format_exc()}{Colors.END}")
        
        print(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
        
        self.save_test_results()
        self.print_summary()
        
        return self.get_critical_failures()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dexy Meeting Agent Installation")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--category", help="Test specific category only")
    
    args = parser.parse_args()
    
    tester = InstallationTester()
    
    if args.category:
        # Run specific category test
        test_method = getattr(tester, f'test_{args.category}', None)
        if test_method:
            tester.print_header()
            test_method()
        else:
            print(f"{Colors.RED}❌ Unknown test category: {args.category}{Colors.END}")
            sys.exit(1)
    else:
        # Run all tests
        critical_failures = tester.run_all_tests()
        
        if critical_failures:
            sys.exit(1)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()