#!/usr/bin/env python3
"""
Simple script to run examples with environment variables loaded
"""

import os
from pathlib import Path

def load_env_file():
    """Load .env file if it exists"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  .env file not found. Create it with your API keys.")
        return False
    
    loaded_keys = []
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                if value and value != f"your-{key.lower().replace('_', '-')}-here":
                    os.environ[key] = value
                    loaded_keys.append(key)
    
    if loaded_keys:
        print(f"✅ Loaded {len(loaded_keys)} environment variables: {loaded_keys}")
        return True
    else:
        print("⚠️  No valid API keys found in .env file")
        return False

def main():
    print("🔧 Loading environment and testing RAG pipeline")
    print("=" * 50)
    
    # Load environment
    env_loaded = load_env_file()
    
    if not env_loaded:
        print("\n📝 To set up API keys:")
        print("1. Edit the .env file with your actual API keys")
        print("2. Get DeepInfra API key from: https://deepinfra.com/dash")
        print("3. Get NCBI API key from: https://www.ncbi.nlm.nih.gov/account/")
        print("\nThen run this script again.")
        return
    
    # Run basic tests
    print(f"\n🧪 Running basic tests...")
    import subprocess
    result = subprocess.run(['python', 'test_basic.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Check if we can run examples
    if os.getenv('DEEPINFRA_API_KEY') or os.getenv('LLM_API_KEY'):
        print(f"\n🚀 API keys loaded! You can now run:")
        print(f"   python example_usage.py")
        print(f"\n📚 Or test individual components:")
        print(f"   python -c \"from core.query_generator import QueryGenerator; print('QueryGenerator ready!')\"")
    else:
        print(f"\n⚠️  API keys still not loaded. Check your .env file.")

if __name__ == "__main__":
    main()