#!/usr/bin/env python3
"""
Setup Environment Script for Dexy Meeting Agent
This script sets up the complete environment for the meeting agent
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import json

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
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class EnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.system_info = self.get_system_info()
        self.required_dirs = [
            'data/audio',
            'data/transcripts', 
            'data/summaries',
            'data/memory',
            'logs'
        ]
        
    def get_system_info(self):
        """Get system information"""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': sys.version_info,
            'is_m1_mac': platform.system() == 'Darwin' and platform.machine() == 'arm64'
        }
    
    def print_header(self):
        """Print setup header"""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 60)
        print("    🤖 DEXY MEETING AGENT ENVIRONMENT SETUP")
        print("=" * 60)
        print(f"{Colors.END}")
        print(f"{Colors.YELLOW}System Info:{Colors.END}")
        print(f"  Platform: {self.system_info['platform']}")
        print(f"  Architecture: {self.system_info['architecture']}")
        print(f"  Python: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}")
        print(f"  M1 Mac: {'Yes' if self.system_info['is_m1_mac'] else 'No'}")
        print()
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print(f"{Colors.BLUE}📋 Checking Python version...{Colors.END}")
        
        if sys.version_info < (3, 8):
            print(f"{Colors.RED}❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}{Colors.END}")
            return False
        
        print(f"{Colors.GREEN}✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible{Colors.END}")
        return True
    
    def create_directories(self):
        """Create required directories"""
        print(f"{Colors.BLUE}📁 Creating project directories...{Colors.END}")
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {dir_path}")
        
        print(f"{Colors.GREEN}✅ All directories created successfully{Colors.END}")
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        print(f"{Colors.BLUE}⚙️ Installing system dependencies...{Colors.END}")
        
        if self.system_info['platform'] == 'Darwin':  # macOS
            self.install_macos_dependencies()
        elif self.system_info['platform'] == 'Linux':
            self.install_linux_dependencies()
        else:
            print(f"{Colors.YELLOW}⚠️ Windows detected. Please install dependencies manually:{Colors.END}")
            print("  - ffmpeg: https://ffmpeg.org/download.html")
            print("  - portaudio: Install via conda or pre-built binaries")
    
    def install_macos_dependencies(self):
        """Install macOS dependencies using Homebrew"""
        if not shutil.which('brew'):
            print(f"{Colors.RED}❌ Homebrew not found. Please install from https://brew.sh/{Colors.END}")
            return False
        
        dependencies = ['ffmpeg', 'portaudio']
        for dep in dependencies:
            try:
                print(f"  Installing {dep}...")
                subprocess.run(['brew', 'install', dep], check=True, capture_output=True)
                print(f"  ✅ {dep} installed")
            except subprocess.CalledProcessError:
                print(f"  ⚠️ {dep} might already be installed or failed to install")
        
        return True
    
    def install_linux_dependencies(self):
        """Install Linux dependencies"""
        dependencies = [
            'ffmpeg',
            'portaudio19-dev',
            'python3-dev',
            'libasound2-dev'
        ]
        
        try:
            print("  Updating package list...")
            subprocess.run(['sudo', 'apt', 'update'], check=True, capture_output=True)
            
            for dep in dependencies:
                print(f"  Installing {dep}...")
                subprocess.run(['sudo', 'apt', 'install', '-y', dep], check=True, capture_output=True)
                print(f"  ✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ Failed to install system dependencies: {e}{Colors.END}")
            return False
        
        return True
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        print(f"{Colors.BLUE}🐍 Setting up virtual environment...{Colors.END}")
        
        venv_path = self.project_root / 'venv'
        
        if venv_path.exists():
            print(f"  ⚠️ Virtual environment already exists at {venv_path}")
            return True
        
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            print(f"  ✅ Virtual environment created at {venv_path}")
            
            # Provide activation instructions
            if self.system_info['platform'] == 'Windows':
                activate_cmd = f"{venv_path}\\Scripts\\activate"
            else:
                activate_cmd = f"source {venv_path}/bin/activate"
            
            print(f"{Colors.YELLOW}  To activate: {activate_cmd}{Colors.END}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ Failed to create virtual environment: {e}{Colors.END}")
            return False
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        print(f"{Colors.BLUE}📦 Installing Python dependencies...{Colors.END}")
        
        # Check if we're in a virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print(f"{Colors.YELLOW}⚠️ Not in virtual environment. Consider activating venv first.{Colors.END}")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            print(f"{Colors.YELLOW}⚠️ requirements.txt not found. Creating basic requirements...{Colors.END}")
            self.create_requirements_file()
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            
            # Install requirements
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)], check=True)
            print(f"{Colors.GREEN}✅ Python dependencies installed successfully{Colors.END}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ Failed to install Python dependencies: {e}{Colors.END}")
            return False
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements = [
            "langchain>=0.1.0",
            "openai>=1.0.0",
            "streamlit>=1.28.0",
            "whisper-openai>=20231117",
            "faiss-cpu>=1.7.4",
            "pydantic>=2.0.0",
            "python-dotenv>=1.0.0",
            "sounddevice>=0.4.6",
            "pyaudio>=0.2.11",
            "pydub>=0.25.1",
            "gtts>=2.4.0",
            "pygame>=2.5.0",
            "speech-recognition>=3.10.0",
            "email-validator>=2.0.0",
            "plotly>=5.17.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "librosa>=0.10.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "chromadb>=0.4.0",
            "tiktoken>=0.5.0",
            "python-multipart>=0.0.6",
            "aiofiles>=23.0.0",
            "watchdog>=3.0.0",
            "psutil>=5.9.0",
            "requests>=2.31.0",
            "beautifulsoup4>=4.12.0",
            "lxml>=4.9.0",
            "python-json-logger>=2.0.0",
            "colorama>=0.4.6",
            "rich>=13.0.0",
            "typer>=0.9.0",
            "uvicorn>=0.23.0",
            "fastapi>=0.104.0"
        ]
        
        # M1 Mac specific requirements
        if self.system_info['is_m1_mac']:
            requirements.extend([
                "tensorflow-macos>=2.13.0",
                "tensorflow-metal>=1.0.0"
            ])
        
        requirements_file = self.project_root / 'requirements.txt'
        with open(requirements_file, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        print(f"  ✅ Created requirements.txt with {len(requirements)} packages")
    
    def create_env_file(self):
        """Create .env file from template"""
        print(f"{Colors.BLUE}⚙️ Setting up environment variables...{Colors.END}")
        
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if env_file.exists():
            print(f"  ⚠️ .env file already exists")
            return True
        
        # Create .env.example if it doesn't exist
        if not env_example.exists():
            self.create_env_example()
        
        # Copy .env.example to .env
        shutil.copy(env_example, env_file)
        print(f"  ✅ Created .env file from template")
        print(f"{Colors.YELLOW}  🔧 Please edit .env file with your API keys and settings{Colors.END}")
        
        return True
    
    def create_env_example(self):
        """Create .env.example file"""
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_FROM=your_email@gmail.com

# Wake Word Configuration
WAKE_WORD=hey dexy
WAKE_WORD_SENSITIVITY=0.5

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
AUDIO_CHANNELS=1

# Memory Configuration
MEMORY_MAX_TOKENS=4000
MEMORY_SIMILARITY_THRESHOLD=0.7

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/dexy_agent.log

# Meeting Configuration
DEFAULT_MEETING_DURATION=60
AUTO_SUMMARY_INTERVAL=300

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
"""
        
        env_example_file = self.project_root / '.env.example'
        with open(env_example_file, 'w') as f:
            f.write(env_content)
        
        print(f"  ✅ Created .env.example file")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        print(f"{Colors.BLUE}📝 Creating .gitignore...{Colors.END}")
        
        gitignore_content = """# Environment files
.env
.env.local
.env.*.local

# Virtual environment
venv/
env/
ENV/

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data files
data/audio/*
data/transcripts/*
data/summaries/*
data/memory/*
!data/audio/.gitkeep
!data/transcripts/.gitkeep
!data/summaries/.gitkeep
!data/memory/.gitkeep

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Streamlit
.streamlit/
"""
        
        gitignore_file = self.project_root / '.gitignore'
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        print(f"  ✅ Created .gitignore file")
    
    def create_gitkeep_files(self):
        """Create .gitkeep files for empty directories"""
        print(f"{Colors.BLUE}📁 Creating .gitkeep files...{Colors.END}")
        
        for dir_path in self.required_dirs:
            if 'data/' in dir_path:
                gitkeep_path = self.project_root / dir_path / '.gitkeep'
                gitkeep_path.touch()
        
        print(f"  ✅ Created .gitkeep files for data directories")
    
    def test_installation(self):
        """Test if installation was successful"""
        print(f"{Colors.BLUE}🧪 Testing installation...{Colors.END}")
        
        try:
            # Test basic imports
            test_imports = [
                'langchain',
                'openai', 
                'streamlit',
                'whisper',
                'faiss',
                'dotenv',
                'sounddevice',
                'pydub',
                'gtts',
                'speech_recognition'
            ]
            
            failed_imports = []
            for module in test_imports:
                try:
                    __import__(module)
                    print(f"  ✅ {module}")
                except ImportError:
                    failed_imports.append(module)
                    print(f"  ❌ {module}")
            
            if failed_imports:
                print(f"{Colors.RED}❌ Some modules failed to import: {failed_imports}{Colors.END}")
                return False
            else:
                print(f"{Colors.GREEN}✅ All critical modules imported successfully{Colors.END}")
                return True
                
        except Exception as e:
            print(f"{Colors.RED}❌ Installation test failed: {e}{Colors.END}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SETUP COMPLETE!{Colors.END}")
        print(f"{Colors.CYAN}Next steps:{Colors.END}")
        print(f"  1. Edit .env file with your API keys")
        print(f"  2. Activate virtual environment: source venv/bin/activate")
        print(f"  3. Run tests: python scripts/test_installation.py")
        print(f"  4. Start the application: streamlit run ui/streamlit_app.py")
        print(f"\n{Colors.YELLOW}Important:{Colors.END}")
        print(f"  - Get OpenAI API key from https://platform.openai.com/")
        print(f"  - Configure email settings for notifications")
        print(f"  - Test microphone permissions before first use")
        print(f"\n{Colors.PURPLE}For help: Check docs/ folder or run with --help{Colors.END}")
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing system dependencies", self.install_system_dependencies),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Setting up environment variables", self.create_env_file),
            ("Creating .gitignore", self.create_gitignore),
            ("Creating .gitkeep files", self.create_gitkeep_files),
            ("Testing installation", self.test_installation)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"{Colors.RED}❌ {step_name} failed: {e}{Colors.END}")
                failed_steps.append(step_name)
        
        if failed_steps:
            print(f"\n{Colors.RED}❌ Setup completed with errors in:{Colors.END}")
            for step in failed_steps:
                print(f"  - {step}")
            print(f"\n{Colors.YELLOW}Please resolve these issues before proceeding.{Colors.END}")
        else:
            self.print_next_steps()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Dexy Meeting Agent Environment")
    parser.add_argument("--skip-system", action="store_true", help="Skip system dependency installation")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    if args.skip_system:
        print(f"{Colors.YELLOW}⚠️ Skipping system dependencies{Colors.END}")
    
    if args.skip_venv:
        print(f"{Colors.YELLOW}⚠️ Skipping virtual environment creation{Colors.END}")
    
    setup.run_setup()

if __name__ == "__main__":
    main()