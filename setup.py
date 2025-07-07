#!/usr/bin/env python3
"""
Setup script for Adobe Resources RAG Chatbot
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists('env_example.txt'):
        shutil.copy('env_example.txt', '.env')
        print("✅ Created .env file from template")
        print("   Please edit .env and add your OpenAI API key")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def check_resources_file():
    """Check if resources.json exists."""
    if os.path.exists('resources.json'):
        print("✅ resources.json found")
        return True
    else:
        print("❌ resources.json not found")
        return False

def run_tests():
    """Run the test script."""
    print("\n🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "test_setup.py"])
        return True
    except subprocess.CalledProcessError:
        print("❌ Tests failed")
        return False

def main():
    """Main setup function."""
    print("🎓 Adobe Resources RAG Chatbot - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check resources file
    if not check_resources_file():
        print("   Please make sure resources.json is in the current directory")
        return False
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("   You can still try running the chatbot, but it may not work properly.")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run the chatbot:")
    print("   - Web version: streamlit run rag_chatbot.py")
    print("   - Command line: python simple_chatbot.py")
    print("3. Or test the setup: python test_setup.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 