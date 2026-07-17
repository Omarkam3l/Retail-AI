import os
import urllib.request

def download_sample():
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample.mp4")
    
    print(f"Downloading sample video from {url}...")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Downloaded successfully! Saved to: {output_path}")

if __name__ == "__main__":
    download_sample()
