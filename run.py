#!/usr/bin/env python3
"""
Ultimate Legal Document Analyzer - Backend Runner
Simple script to start the FastAPI application with proper configuration.
"""

import os
import sys
import uvicorn
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📋 Please copy .env.example to .env and configure your API keys")
        print("   cp .env.example .env")
        return False

    # Check if GEMINI_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in .env file!")
        print("📋 Please add your Gemini API key to the .env file")
        return False

    print("✅ Environment configuration looks good!")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import torch
        import transformers
        import chromadb
        import httpx
        import PyPDF2
        print("✅ All required dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install dependencies: pip install -r requirements.txt")
        return False

def main():
    """Main function to start the application"""
    print("🚀 Starting Ultimate Legal Document Analyzer Backend...")
    print("=" * 60)

    # Change to backend directory if not already there
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)

    # Check environment and dependencies
    if not check_environment():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    print(f"🌐 Server will start on: http://{host}:{port}")
    print(f"📊 Workers: {workers}")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print(f"📝 Log level: {log_level}")
    print("=" * 60)
    print("📚 Legal Models will be loaded on first request...")
    print("🔍 ChromaDB will initialize automatically...")
    print("🤖 Gemini AI integration is ready...")
    print("=" * 60)

    try:
        # Start the server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,  # Workers=1 when reload=True
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()