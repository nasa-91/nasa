import numpy as np
import pandas as pd

np.random.seed(2024)

T = 888
years = np.linspace(1950, 1950 + T/12, T)

base_signal = 0.15 * np.sin(2*np.pi*years/50) + 0.08 * np.sin(2*np.pi*years/25)
seasonal = 0.06 * np.sin(2*np.pi*(years-1950))
noise = np.random.normal(0, 0.5, T)
nino34 = base_signal + seasonal + noise

el_nino_events = [
    (57, 72, 2.2),
    (126, 144, 1.8),
    (215, 234, 2.5),
    (348, 369, 2.8),
    (438, 459, 2.6),
    (564, 585, 2.9),
    (684, 705, 2.4),
    (792, 813, 2.7)
]

for start, end, intensity in el_nino_events:
    if end < T:
        center = (start + end) // 2
        for i in range(start, end):
            dist = abs(i - center) / ((end-start)/2)
            nino34[i] += intensity * np.exp(-dist**2 / 0.35)

la_nina_events = [
    (84, 108, -1.8),
    (168, 192, -2.1),
    (270, 294, -1.9),
    (396, 420, -2.4),
    (510, 534, -2.0),
    (618, 642, -2.3),
    (738, 762, -2.2),
    (840, 864, -1.95)
]

for start, end, intensity in la_nina_events:
    if end < T:
        center = (start + end) // 2
        for i in range(start, end):
            dist = abs(i - center) / ((end-start)/2)
            nino34[i] += intensity * np.exp(-dist**2 / 0.35)

dates = pd.date_range(start='1950-01-01', periods=T, freq='MS')

df = pd.DataFrame({
    'date': dates.strftime('%Y-%m'),
    'nino34': np.round(nino34, 2)
})

output_path = 'venv_env/data_nino.csv'
df.to_csv(output_path, index=False)

print("="*60)
print("Sample Data Generated")
print("="*60)
print(f"\nOutput: {output_path}")
print(f"\nStatistics:")
print(f"  Period: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
print(f"  Observations: {len(df)}")
print(f"  Span: {(T//12)} years {(T%12)} months")
print(f"\nNINO3.4 Index:")
print(f"  Mean:   {df['nino34'].mean():+.3f} C")
print(f"  Std:    {df['nino34'].std():.3f} C")
print(f"  Min:    {df['nino34'].min():.2f} C (La Nina)")
print(f"  Max:    {df['nino34'].max():.2f} C (El Nino)")
print(f"\nExtreme Events:")
el_nino_count = len(df[df['nino34'] > 1.5])
la_nina_count = len(df[df['nino34'] < -1.5])
print(f"  Strong El Nino (>+1.5C): {el_nino_count} months")
print(f"  Strong La Nina (<-1.5C): {la_nina_count} months")

print(f"\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print(f"\nLast 10 rows:")
print(df.tail(10).to_string(index=False))

print("\n" + "="*60)
print("[OK] Data ready for analysis")
print("="*60)
