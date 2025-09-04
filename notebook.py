import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    return pd,


@app.cell
def __(pd):
    # Import nussgipfel sales data with proper date parsing
    nussgipfel = pd.read_csv('data/sales/nussgipfel.csv', date_format="%Y-%m-%d", parse_dates=['date'])
    
    # Import weather data with proper date parsing and separator
    weather = pd.read_csv('data/climate/nbcn-daily_BER_previous.csv', sep=';', date_format="%Y%m%d", parse_dates=['date'])
    
    return nussgipfel, weather


@app.cell
def __(pd, nussgipfel, weather):
    # Select weather variables most likely to correlate with pastry sales
    # Based on the documentation in 1_how-to-download-nbcn-d.txt
    weather_relevant = weather[['date', 'tre200d0', 'rre150d0', 'sre000d0', 'nto000d0', 'ure200d0']].copy()
    
    # Rename columns to be more readable
    weather_relevant.columns = ['date', 'avg_temp', 'precipitation', 'sunshine_duration', 'cloud_coverage', 'humidity']
    
    # Merge with sales data
    merged = pd.merge(nussgipfel, weather_relevant, on='date', how='inner')
    
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print("\nSelected weather variables:")
    print("- avg_temp: Average temperature (°C)")
    print("- precipitation: Daily rainfall (mm)")
    print("- sunshine_duration: Sunshine duration (minutes)")
    print("- cloud_coverage: Cloud coverage (%)")
    print("- humidity: Relative humidity (%)")
    print("\nMerged data sample:")
    print(merged.head())
    
    return merged


if __name__ == "__main__":
    app.run()
