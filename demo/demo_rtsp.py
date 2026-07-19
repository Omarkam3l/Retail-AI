"""Demo: Run pipeline on RTSP stream."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run AI pipeline on RTSP stream")
    parser.add_argument("--url", required=True, help="RTSP stream URL")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    print(f"Connecting to RTSP stream: {args.url}")
    cmd = f'python scripts/run_demo.py --video "{args.url}"'
    if args.save:
        cmd += f' --save "{args.save}"'
    os.system(cmd)

if __name__ == "__main__":
    main()
