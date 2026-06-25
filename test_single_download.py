import pandas as pd
import requests

df = pd.read_csv('data/raw/fitzpatrick17k/fitzpatrick17k.csv')
sample = df.head(5)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for _, row in sample.iterrows():
    url = row['url']
    print(f"\nTrying: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        print(f"  Status: {response.status_code}")
        print(f"  Content length: {len(response.content)} bytes")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")