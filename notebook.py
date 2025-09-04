import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    # Set pandas display options for better table formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    return (pd,)


@app.cell
def _(pd):
    # Import nussgipfel sales data with proper date parsing
    nussgipfel = pd.read_csv('data/sales/nussgipfel.csv', date_format="%Y-%m-%d", parse_dates=['date'])

    # Import weather data with proper date parsing and separator
    weather = pd.read_csv('data/climate/nbcn-daily_BER_previous.csv', sep=';', date_format="%Y%m%d", parse_dates=['date'], na_values='-')
    return nussgipfel, weather


@app.cell
def _(nussgipfel, pd, weather):
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
    return (merged,)


@app.cell
def _(merged):
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
    return (df_weather,)


@app.cell
def _(df_weather):
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
    return (enhanced_data,)


@app.cell
def _(enhanced_data):
    # WEATHER vs NUSSGIPFEL SALES: CORRELATION ANALYSIS
    print("WEATHER vs NUSSGIPFEL SALES: CORRELATION ANALYSIS")
    print("="*55)

    print(f"Dataset: {len(enhanced_data)} days from {enhanced_data['date'].min().strftime('%Y-%m-%d')} to {enhanced_data['date'].max().strftime('%Y-%m-%d')}")
    print()

    # Calculate both daily and monthly correlations
    daily_correlation = enhanced_data['weather_goodness'].corr(enhanced_data['amount'])

    # Create monthly averages for monthly correlation (using unique variable names)
    corr_monthly_data = enhanced_data.copy()
    corr_monthly_data['year_month'] = corr_monthly_data['date'].dt.to_period('M')
    corr_monthly_avg = corr_monthly_data.groupby('year_month').agg({
        'amount': 'mean',
        'weather_goodness': 'mean'
    }).reset_index()
    monthly_correlation = corr_monthly_avg['amount'].corr(corr_monthly_avg['weather_goodness'])

    print("CORRELATIONS:")
    print(f"Daily correlation:   {daily_correlation:.3f}")
    print(f"Monthly correlation: {monthly_correlation:.3f}")
    print()

    print("EXPLANATION:")
    print("• Daily correlation: Compares weather goodness vs sales on individual days")
    print("  - Measures immediate weather impact (e.g., rainy Tuesday vs sunny Tuesday)")
    print("  - Includes daily noise (weekends, holidays, random events)")
    print()
    print("• Monthly correlation: Compares average weather vs average sales by month")
    print("  - Measures seasonal patterns (e.g., sunny July vs rainy January)")
    print("  - Smooths out daily noise, reveals longer-term trends")
    print("  - Shows if months with better weather generally have higher sales")
    print()

    # Interpret the stronger correlation
    stronger_corr = max(abs(daily_correlation), abs(monthly_correlation))
    if abs(monthly_correlation) > abs(daily_correlation):
        main_correlation = monthly_correlation
        timeframe = "monthly"
        print("→ Monthly correlation is STRONGER")
        print("  Weather patterns matter more over seasonal trends than daily variations")
    else:
        main_correlation = daily_correlation
        timeframe = "daily"
        print("→ Daily correlation is STRONGER")
        print("  Immediate weather effects matter more than seasonal patterns")

    print()

    # Interpret the correlation strength
    if abs(main_correlation) < 0.1:
        strength = "Very weak"
    elif abs(main_correlation) < 0.3:
        strength = "Weak"
    elif abs(main_correlation) < 0.5:
        strength = "Moderate"
    elif abs(main_correlation) < 0.7:
        strength = "Strong"
    else:
        strength = "Very strong"

    direction = "positive" if main_correlation > 0 else "negative"

    print(f"INTERPRETATION: {strength} {direction} correlation ({timeframe})")
    print()

    if main_correlation > 0:
        print("✓ Better weather IS associated with higher nussgipfel sales")
    else:
        print("✗ Better weather is NOT associated with higher nussgipfel sales")

    print()
    print("What this means:")
    if abs(main_correlation) < 0.2:
        print("- Weather has minimal impact on nussgipfel sales")
        print("- Other factors likely drive sales more than weather")
    elif abs(main_correlation) < 0.4:
        print("- Weather has a noticeable but moderate impact on sales")
        print("- Weather explains some variation in sales, but other factors matter too")
    else:
        print("- Weather has a substantial impact on nussgipfel sales")
        print("- Weather is an important factor in predicting sales")

    # Show some concrete examples
    print()
    print("CONCRETE EXAMPLES:")

    # Best weather days
    best_weather = enhanced_data.nlargest(5, 'weather_goodness')
    worst_weather = enhanced_data.nsmallest(5, 'weather_goodness')

    print(f"Best weather days (avg score {best_weather['weather_goodness'].mean():.1f}): avg sales = {best_weather['amount'].mean():.1f}")
    print(f"Worst weather days (avg score {worst_weather['weather_goodness'].mean():.1f}): avg sales = {worst_weather['amount'].mean():.1f}")

    return


@app.cell
def _(enhanced_data):
    # Create a cleaner chart with normalized values and monthly averages
    print("MONTHLY AVERAGE NUSSGIPFEL SALES & WEATHER GOODNESS")
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
    import matplotlib.pyplot as plt
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
    return


if __name__ == "__main__":
    app.run()
