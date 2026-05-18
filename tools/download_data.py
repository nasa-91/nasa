import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
import os


def download_noaa_nino34():
    print("=" * 60)
    print("NOAA NINO3.4 Data Download Tool")
    print("=" * 60)

    output_dir = 'venv_env'
    os.makedirs(output_dir, exist_ok=True)

    url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.asc"

    temp_file = os.path.join(output_dir, 'nino34_raw.txt')
    csv_file = os.path.join(output_dir, 'data_nino.csv')

    print(f"\nDownloading data...")
    print(f"Source: {url}")

    try:
        urllib.request.urlretrieve(url, temp_file)
        print(f"[OK] Downloaded: {temp_file}")

        print(f"\nParsing data...")
        with open(temp_file, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        data_rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    nino34_value = float(parts[6])

                    date_str = f"{year}-{month:02d}-01"
                    data_rows.append([date_str, nino34_value])

                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(data_rows, columns=['date', 'nino34'])

        df.to_csv(csv_file, index=False)

        print(f"\n[OK] Parsing complete")
        print(f"\nStatistics:")
        print(f"  Period: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"  Observations: {len(df)}")
        print(f"  Mean: {df['nino34'].mean():.3f} C")
        print(f"  Std: {df['nino34'].std():.3f} C")
        print(f"  Min: {df['nino34'].min():.2f} C (La Nina)")
        print(f"  Max: {df['nino34'].max():.2f} C (El Nino)")

        missing = df['nino34'].isna().sum()
        if missing > 0:
            print(f"  [WARN] Missing values: {missing} ({missing/len(df)*100:.1f}%)")
        else:
            print(f"  [OK] No missing values")

        print(f"\n[OK] CSV saved: {csv_file}")

        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string(index=False))

        print(f"\nLast 10 rows:")
        print(df.tail(10).to_string(index=False))

        return csv_file

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nFallback: Using sample data")
        return None


def create_sample_data():
    """Generate sample data if download fails"""
    print("\nGenerating synthetic NINO3.4 data...")

    np.random.seed(42)
    T = 900
    t = np.arange(T)

    base_signal = 0.25 * np.sin(2*np.pi*t/48) + 0.15 * np.sin(2*np.pi*t/24)
    seasonal = 0.08 * np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 0.55, T)
    synthetic = base_signal + seasonal + noise

    event_peaks = [
        (120, 140), (280, 305), (430, 455),
        (580, 610), (730, 760)
    ]
    for start, end in event_peaks:
        if end < T:
            intensity = np.random.uniform(2.0, 3.0)
            center = (start + end) // 2
            for i in range(start, end):
                dist = abs(i - center) / ((end-start)/2)
                synthetic[i] += intensity * np.exp(-dist**2 / 0.4)

    trough_peaks = [
        (170, 195), (330, 355), (490, 520),
        (640, 670), (800, 830)
    ]
    for start, end in trough_peaks:
        if end < T:
            intensity = np.random.uniform(-2.0, -2.8)
            center = (start + end) // 2
            for i in range(start, end):
                dist = abs(i - center) / ((end-start)/2)
                synthetic[i] += intensity * np.exp(-dist**2 / 0.4)

    dates = pd.date_range(start='1950-01-01', periods=T, freq='MS')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'nino34': synthetic
    })

    output_path = 'venv_env/data_nino.csv'
    df.to_csv(output_path, index=False)

    print(f"[OK] Sample data generated: {output_path}")
    print(f"  Contains {T} months (1950-01 to 2024-12)")
    print(f"  Note: This is synthetic data, replace with real NOAA data for analysis")

    return output_path


if __name__ == '__main__':
    result = download_noaa_nino34()

    if result is None:
        result = create_sample_data()

    print("\n" + "="*60)
    print("Next step:")
    print("  Run analysis: python enso_advanced_analysis.py")
    print("="*60)
