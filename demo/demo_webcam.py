"""Demo: Run pipeline on webcam feed."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    print("Starting webcam pipeline demo...")
    print("Press 'q' to quit.")
    os.system('python scripts/run_demo.py --video 0')

if __name__ == "__main__":
    main()
