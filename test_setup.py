#!/usr/bin/env python3
"""
Test script to verify the Adobe Resources RAG chatbot setup.
"""

import json
import os
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ openai import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError as e:
        print(f"❌ chromadb import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence_transformers import failed: {e}")
        return False
    
    try:
        import tiktoken
        print("✅ tiktoken imported successfully")
    except ImportError as e:
        print(f"❌ tiktoken import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

def test_resources_file():
    """Test if resources.json file exists and can be loaded."""
    print("\nTesting resources.json file...")
    
    if not os.path.exists('resources.json'):
        print("❌ resources.json file not found")
        return False
    
    try:
        with open('resources.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ resources.json loaded successfully")
        print(f"   - Contains {len(data)} resources")
        
        # Show some sample data
        sample_keys = list(data.keys())[:3]
        print(f"   - Sample resources: {sample_keys}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading resources.json: {e}")
        return False

def test_environment():
    """Test environment variables."""
    print("\nTesting environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        print("✅ OPENAI_API_KEY found")
        return True
    else:
        print("❌ OPENAI_API_KEY not found or not set")
        print("   Please set your OpenAI API key in the .env file")
        return False

def test_sentence_transformer():
    """Test if sentence transformer can be loaded."""
    print("\nTesting sentence transformer...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading sentence transformer: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Adobe Resources RAG Chatbot - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_resources_file,
        test_environment,
        test_sentence_transformer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo run the chatbot:")
        print("  streamlit run rag_chatbot.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Set your OpenAI API key in the .env file")
        print("  3. Make sure resources.json is in the current directory")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 