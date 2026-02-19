import pandas as pd
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = 'data/raw/fitzpatrick17k/fitzpatrick17k.csv'
SAVE_DIR = 'data/raw/fitzpatrick17k/images'
SAMPLE_SIZE = 1000  # Change to None if you want the full 16,577 images later
MAX_WORKERS = 10    # Adjust based on your internet stability (5-20)

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(row):
    """
    Downloads a single image from the URL in the CSV row.
    Uses the md5hash as the filename to maintain linkage.
    """
    url = row['url']
    file_name = f"{row['md5hash']}.jpg"
    file_path = os.path.join(SAVE_DIR, file_name)
    
    # 1. Skip if already exists (Resume support)
    if os.path.exists(file_path):
        return "skipped"

    # 2. Attempt download
    try:
        # User-agent mimics a browser to avoid getting blocked by servers
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return "success"
        else:
            return f"failed_status_{response.status_code}"
    except Exception as e:
        return f"error_{str(e)}"

def run_tiny_bedrock_download():
    # Load metadata
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Could not find {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Stratified Sampling: Ensures we get a mix of different skin types and labels
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"⚖️  Sampling {SAMPLE_SIZE} images for the Tiny Bedrock...")
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df

    print(f"🚀 Starting download of {len(df_sample)} images to: {SAVE_DIR}")

    # ThreadPoolExecutor for concurrent downloads
    results = {"success": 0, "skipped": 0, "failed": 0}
    
    with tqdm(total=len(df_sample), desc="Downloading", unit="img") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map the rows to the download function
            futures = [executor.submit(download_image, row) for _, row in df_sample.iterrows()]
            
            for future in as_completed(futures):
                res = future.result()
                if res == "success":
                    results["success"] += 1
                elif res == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
                
                pbar.update(1)

    print("\n" + "="*30)
    print("✅ DOWNLOAD COMPLETE")
    print(f"📦 Total Images: {len(df_sample)}")
    print(f"🟢 Success:      {results['success']}")
    print(f"🟡 Skipped:      {results['skipped']}")
    print(f"🔴 Failed:       {results['failed']}")
    print("="*30)

if __name__ == "__main__":
    run_tiny_bedrock_download()