#!/usr/bin/env python3
"""
Local test script to verify the app works before deployment
"""
import subprocess
import sys
import time
import requests
import threading

def test_app():
    """Test the Flask app locally"""
    print("=== Testing KrishiMitra App Locally ===")
    
    # Start the Flask app in a separate process
    print("Starting Flask app...")
    process = subprocess.Popen([
        sys.executable, "app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait a moment for the app to start
    time.sleep(3)
    
    try:
        # Test the endpoints
        print("Testing endpoints...")
        
        # Test home endpoint
        try:
            response = requests.get("http://localhost:5000/", timeout=5)
            print(f"✅ Home endpoint: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"❌ Home endpoint failed: {e}")
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            print(f"✅ Health endpoint: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
    
    finally:
        # Clean up
        print("Stopping Flask app...")
        process.terminate()
        process.wait()
        
        # Print any output
        stdout, stderr = process.communicate()
        if stdout:
            print("STDOUT:", stdout)
        if stderr:
            print("STDERR:", stderr)

if __name__ == "__main__":
    test_app()
