"""
Test script to verify the Streamlit app can run properly
"""

import subprocess
import time
import sys
import signal
import os


def test_streamlit_app():
    """Test that the Streamlit app can start without errors."""
    
    print("🧪 Testing Streamlit App Startup")
    print("=" * 40)
    
    try:
        # Start the Streamlit app in a subprocess
        print("🚀 Starting Streamlit app...")
        
        # Use subprocess to run streamlit
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8502"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds for the app to start
        print("⏳ Waiting for app to initialize...")
        time.sleep(5)
        
        # Check if the process is still running (no immediate crash)
        if process.poll() is None:
            print("✅ Streamlit app started successfully!")
            print("✅ No immediate startup errors detected")
            
            # Try to get some output
            try:
                stdout, stderr = process.communicate(timeout=2)
                if "You can now view your Streamlit app" in stdout or "Local URL" in stdout:
                    print("✅ App is serving on local URL")
            except subprocess.TimeoutExpired:
                # This is expected - the app should keep running
                print("✅ App is running continuously (as expected)")
            
            result = True
        else:
            # Process terminated - check for errors
            stdout, stderr = process.communicate()
            print("❌ Streamlit app failed to start")
            print(f"Exit code: {process.returncode}")
            if stderr:
                print(f"Error output: {stderr}")
            result = False
        
        # Clean up - terminate the process if it's still running
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing Streamlit app: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_streamlit_app()
    
    if success:
        print("\n🎉 Integration test PASSED!")
        print("\nThe dashboard is ready to use. To run it:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Integration test FAILED!")
        sys.exit(1)