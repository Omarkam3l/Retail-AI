"""Demo: Run pipeline on a local video file."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run AI pipeline on video file")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    print(f"Processing video: {args.video}")
    print(f"Confidence threshold: {args.conf}")
    print("Use scripts/run_demo.py for full pipeline with visualization.")

    # Delegate to the main demo script
    os.system(f'python scripts/run_demo.py --video "{args.video}" --conf {args.conf}' +
              (f' --save "{args.save}"' if args.save else ''))

if __name__ == "__main__":
    main()
