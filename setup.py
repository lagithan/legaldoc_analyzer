#!/usr/bin/env python3
"""
Ultimate Legal Document Analyzer - Setup Script
Automated setup for the backend application.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    if Path("venv").exists():
        print("📁 Virtual environment already exists")
        return True

    print("🐍 Creating virtual environment...")
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False

    print("✅ Virtual environment created successfully!")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")

    # Determine the correct pip command for the virtual environment
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        pip_cmd = "venv/bin/pip"

    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False

    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        return False

    print("✅ Dependencies installed successfully!")
    return True

def setup_environment_file():
    """Set up environment configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print("📄 .env file already exists")
        return True

    if not env_example.exists():
        print("❌ .env.example file not found!")
        return False

    # Copy example to actual env file
    shutil.copy(env_example, env_file)
    print("📄 Created .env file from template")
    print("⚠️  Please edit .env file and add your GEMINI_API_KEY")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "chroma_db",
        "logs",
        "temp"
    ]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")

    return True

def check_gpu_support():
    """Check for GPU support"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU Support: Available ({gpu_count} GPU(s))")
            print(f"   Primary GPU: {gpu_name}")
        else:
            print("💻 GPU Support: Not available (will use CPU)")
    except ImportError:
        print("❓ GPU Support: Cannot check (PyTorch not installed yet)")

def main():
    """Main setup function"""
    print("🚀 Ultimate Legal Document Analyzer - Setup")
    print("=" * 60)

    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Setup virtual environment
    if not setup_virtual_environment():
        sys.exit(1)

    # Step 3: Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Step 4: Setup environment file
    if not setup_environment_file():
        sys.exit(1)

    # Step 5: Create directories
    if not create_directories():
        sys.exit(1)

    # Step 6: Check GPU support
    check_gpu_support()

    print("=" * 60)
    print("✅ Setup completed successfully!")
    print("")
    print("📋 Next steps:")
    print("   1. Edit .env file and add your GEMINI_API_KEY")
    print("   2. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("      venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print("      source venv/bin/activate")
    print("   3. Start the application:")
    print("      python run.py")
    print("")
    print("🌐 The server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()