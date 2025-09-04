import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    
    # Set pandas display options for better table formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    return pd,


@app.cell
def __(pd):
    # Import nussgipfel sales data with proper date parsing
    nussgipfel = pd.read_csv('data/sales/nussgipfel.csv', date_format="%Y-%m-%d", parse_dates=['date'])
    
    # Import weather data with proper date parsing and separator
    weather = pd.read_csv('data/climate/nbcn-daily_BER_previous.csv', sep=';', date_format="%Y%m%d", parse_dates=['date'], na_values='-')
    
    return nussgipfel, weather


@app.cell
def __(pd, nussgipfel, weather):
    # Select weather variables most likely to correlate with pastry sales
    # Based on the documentation in 1_how-to-download-nbcn-d.txt
    weather_relevant = weather[['date', 'tre200d0', 'rre150d0', 'sre000d0', 'ure200d0']].copy()
    
    # Rename columns to be more readable
    weather_relevant.columns = ['date', 'avg_temp', 'precipitation', 'sunshine_duration', 'humidity']
    
    # Clean the data: convert '-' to NaN and then to numeric
    weather_cols = ['avg_temp', 'precipitation', 'sunshine_duration', 'humidity']
    for col in weather_cols:
        weather_relevant[col] = pd.to_numeric(weather_relevant[col], errors='coerce')
    
    # Merge with sales data
    merged = pd.merge(nussgipfel, weather_relevant, on='date', how='inner')

    print(f"Merged dataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print("\nSelected weather variables:")
    print("- avg_temp: Average temperature (°C)")
    print("- precipitation: Daily rainfall (mm)")
    print("- sunshine_duration: Sunshine duration (minutes)")
    print("- humidity: Relative humidity (%)")
    print("\nData cleaning applied: '-' values converted to NaN")
    print("\nMerged data sample:")
    print(merged.head())

    return merged


@app.cell
def __(merged):
    # Simple correlation analysis: Do weather variables correlate with nussgipfel sales?
    print("CORRELATION ANALYSIS")
    print("=" * 40)

    # Calculate correlations between amount (sales) and weather variables
    weather_variables = ['avg_temp', 'precipitation', 'sunshine_duration', 'humidity']

    print("Correlation between nussgipfel sales (amount) and weather:")
    for variable in weather_variables:
        corr = merged['amount'].corr(merged[variable])
        print(f"{variable:20}: {corr:6.3f}")

    print("\nInterpretation:")
    print("- Values close to +1: Strong positive correlation")
    print("- Values close to -1: Strong negative correlation")
    print("- Values close to  0: No correlation")

    return


@app.cell
def __(merged, pd):
    # Create a weather goodness score (1-10) based on intuitive weather quality
    print("CREATING WEATHER GOODNESS SCORE")
    print("="*40)
    
    df_weather = merged.copy()
    
    # Define what makes "good weather" for each variable
    # Temperature: 15-25°C is ideal (comfortable for walking/shopping)
    temp_ideal = 20  # ideal temperature
    temp_range = 10  # acceptable range around ideal
    temp_score = 1 - abs(df_weather['avg_temp'] - temp_ideal) / temp_range
    temp_score = temp_score.clip(0, 1)  # keep between 0-1
    
    # Sunshine: more minutes = better (normalize to max observed)
    sunshine_score = df_weather['sunshine_duration'] / df_weather['sunshine_duration'].max()
    
    # Precipitation: less rain = better (0mm = perfect, normalize to reasonable max)
    precip_max = 20  # anything above 20mm is considered very rainy
    precip_score = 1 - (df_weather['precipitation'] / precip_max).clip(0, 1)
    
    # Humidity: 40-60% is comfortable, too high or low is bad
    humidity_ideal = 50
    humidity_range = 30
    humidity_score = 1 - abs(df_weather['humidity'] - humidity_ideal) / humidity_range
    humidity_score = humidity_score.clip(0, 1)
    
    # Combine with intuitive weights (you can adjust these)
    weights = {
        'sunshine': 0.4,      # Most important - sunny days encourage outings
        'temperature': 0.3,   # Important - comfortable temperature
        'precipitation': 0.2, # Important - no rain
        'humidity': 0.1       # Less important - comfort factor
    }
    
    weather_goodness = (
        weights['sunshine'] * sunshine_score +
        weights['temperature'] * temp_score +
        weights['precipitation'] * precip_score +
        weights['humidity'] * humidity_score
    )
    
    # Scale to 1-10
    weather_goodness_score = 1 + 9 * weather_goodness
    
    df_weather['weather_goodness'] = weather_goodness_score
    
    print("Weights used:")
    for key, value in weights.items():
        print(f"  {key}: {value}")
    
    print(f"\nWeather Goodness Score range: {weather_goodness_score.min():.1f} to {weather_goodness_score.max():.1f}")
    print(f"Weather Goodness Score mean: {weather_goodness_score.mean():.1f}")
    
    # Correlation with sales
    goodness_corr = merged['amount'].corr(weather_goodness_score)
    print(f"\nCORRELATION: Weather Goodness vs Nussgipfel Sales: {goodness_corr:.3f}")
    
    print(f"\nSample data:")
    print(df_weather[['date', 'amount', 'avg_temp', 'sunshine_duration', 'precipitation', 'humidity', 'weather_goodness']].head())
    
    return df_weather, weather_goodness_score


@app.cell
def __(df_weather):
    # Enhanced data table with weather goodness score
    print("ENHANCED DATA WITH WEATHER GOODNESS SCORE")
    print("="*50)
    
    # Select relevant columns for display
    enhanced_data = df_weather[['id', 'date', 'article', 'amount', 'avg_temp', 'precipitation', 'sunshine_duration', 'humidity', 'weather_goodness']].copy()
    
    # Round weather goodness for cleaner display
    enhanced_data['weather_goodness'] = enhanced_data['weather_goodness'].round(1)
    
    print("Enhanced merged data sample:")
    print(enhanced_data.head())
    
    print(f"\nData summary:")
    print(f"Total records: {len(enhanced_data)}")
    print(f"Weather goodness range: {enhanced_data['weather_goodness'].min():.1f} - {enhanced_data['weather_goodness'].max():.1f}")
    print(f"Average weather goodness: {enhanced_data['weather_goodness'].mean():.1f}")
    
    return enhanced_data


@app.cell
def __(enhanced_data):
    import matplotlib.pyplot as plt

    # Create a cleaner chart with normalized values and monthly averages
    print("CLEAN COMPARISON: NUSSGIPFEL SALES & WEATHER GOODNESS")
    print("="*55)
    
    # Create monthly averages for smoother visualization
    monthly_data = enhanced_data.copy()
    monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
    monthly_avg = monthly_data.groupby('year_month').agg({
        'amount': 'mean',
        'weather_goodness': 'mean'
    }).reset_index()
    monthly_avg['date'] = monthly_avg['year_month'].dt.to_timestamp()
    
    # Normalize both metrics to 0-100 scale for comparison
    sales_norm = ((monthly_avg['amount'] - monthly_avg['amount'].min()) / 
                  (monthly_avg['amount'].max() - monthly_avg['amount'].min())) * 100
    
    weather_norm = ((monthly_avg['weather_goodness'] - monthly_avg['weather_goodness'].min()) / 
                    (monthly_avg['weather_goodness'].max() - monthly_avg['weather_goodness'].min())) * 100
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(monthly_avg['date'], sales_norm, 'b-', label='Nussgipfel Sales (normalized)', linewidth=3, marker='o', markersize=4)
    ax.plot(monthly_avg['date'], weather_norm, 'r-', label='Weather Goodness (normalized)', linewidth=3, marker='s', markersize=4)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Value (0-100)', fontsize=12)
    ax.set_title('Monthly Averages: Nussgipfel Sales vs Weather Goodness\n(Both normalized to 0-100 scale)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Show correlation on monthly data
    monthly_corr = monthly_avg['amount'].corr(monthly_avg['weather_goodness'])
    print(f"\nMonthly correlation: {monthly_corr:.3f}")
    print(f"Original daily correlation: {enhanced_data['amount'].corr(enhanced_data['weather_goodness']):.3f}")
    
    return monthly_avg


if __name__ == "__main__":
    app.run()
