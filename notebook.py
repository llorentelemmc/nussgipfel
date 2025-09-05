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
def _():
    return


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

    print("CORRELATION COEFFICIENTS:")
    print(f"Daily:   {daily_correlation:.3f}")
    print(f"Monthly: {monthly_correlation:.3f}")
    print()

    print("Daily correlation coefficient: 0.267")
    print("This is a weak to moderate positive correlation")
    print("On any given day, there's a weak tendency for better weather to coincide with higher sales")
    print("Only explains about 7% of the variance (0.267² ≈ 0.07)")
    print()
    print("Monthly correlation coefficient: 0.517")
    print("This is a moderate to strong positive correlation")
    print("When you average out daily fluctuations to monthly patterns, the weather-sales relationship becomes much clearer")
    print("Explains about 27% of the variance (0.517² ≈ 0.27)")
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
    import matplotlib.pyplot as viz_plt
    viz_fig, viz_ax = viz_plt.subplots(figsize=(12, 6))

    viz_ax.plot(monthly_avg['date'], sales_norm, 'b-', label='Nussgipfel Sales (normalized)', linewidth=3, marker='o', markersize=4)
    viz_ax.plot(monthly_avg['date'], weather_norm, 'r-', label='Weather Goodness (normalized)', linewidth=3, marker='s', markersize=4)

    viz_ax.set_xlabel('Date', fontsize=12)
    viz_ax.set_ylabel('Normalized Value (0-100)', fontsize=12)
    viz_ax.set_title('Monthly Averages: Nussgipfel Sales vs Weather Goodness\n(Both normalized to 0-100 scale)', fontsize=14)
    viz_ax.legend(fontsize=11)
    viz_ax.grid(True, alpha=0.3)

    viz_plt.xticks(rotation=45)
    viz_plt.tight_layout()
    viz_plt.show()
    return


@app.cell
def _(enhanced_data):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # MONTH-BASED SALES FORECASTING MODEL
    print("MONTH-BASED SALES FORECASTING MODEL")
    print("=" * 40)

    # Prepare monthly data for modeling
    pred_monthly_data = enhanced_data.copy()
    pred_monthly_data['year_month'] = pred_monthly_data['date'].dt.to_period('M')
    pred_monthly_data['month'] = pred_monthly_data['date'].dt.month
    pred_monthly_data['year'] = pred_monthly_data['date'].dt.year

    pred_monthly_avg = pred_monthly_data.groupby(['year_month', 'month', 'year']).agg({
        'amount': 'mean',
        'weather_goodness': 'mean'
    }).reset_index()
    pred_monthly_avg['date'] = pred_monthly_avg['year_month'].dt.to_timestamp()

    # Create month dummy variables (one-hot encoding)
    month_dummies = np.zeros((len(pred_monthly_avg), 12))
    for i, month in enumerate(pred_monthly_avg['month']):
        month_dummies[i, month - 1] = 1  # month-1 because months are 1-12, array is 0-11

    # Prepare features: months + year trend
    years_since_start = pred_monthly_avg['year'] - pred_monthly_avg['year'].min()
    X = np.column_stack([month_dummies, years_since_start])
    y = pred_monthly_avg['amount'].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    pred_monthly_avg['predicted_sales'] = y_pred

    # Calculate model performance
    r2 = r2_score(y, y_pred)

    print(f"Model Performance:")
    print(f"R² Score: {r2:.3f} ({r2 * 100:.1f}% of variance explained)")
    print()

    # Show seasonal effects
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print("Seasonal Effects (relative to baseline):")
    for i, month_name in enumerate(month_names):
        effect = model.coef_[i]
        print(f"{month_name}: {effect:+.1f} sales")

    print()
    year_trend = model.coef_[-1]  # Last coefficient is year trend
    print(f"Year-over-year growth: {year_trend:+.1f} sales per year")
    return (pred_monthly_avg,)


@app.cell
def _(enhanced_data, pred_monthly_avg):
    # Create a cleaner chart with normalized values and monthly averages
    print("MONTHLY SALES: ACTUAL vs SEASONAL FORECAST MODEL")
    print("="*55)

    # Create monthly averages for smoother visualization
    final_monthly_data = enhanced_data.copy()
    final_monthly_data['year_month'] = final_monthly_data['date'].dt.to_period('M')
    final_monthly_avg = final_monthly_data.groupby('year_month').agg({
        'amount': 'mean',
        'weather_goodness': 'mean'
    }).reset_index()
    final_monthly_avg['date'] = final_monthly_avg['year_month'].dt.to_timestamp()

    # Normalize sales for comparison (both actual and predicted on same scale)
    final_sales_norm = ((final_monthly_avg['amount'] - final_monthly_avg['amount'].min()) / 
                       (final_monthly_avg['amount'].max() - final_monthly_avg['amount'].min())) * 100

    # Normalize predicted sales using the same scale as actual sales
    final_pred_norm = ((pred_monthly_avg['predicted_sales'] - final_monthly_avg['amount'].min()) / 
                      (final_monthly_avg['amount'].max() - final_monthly_avg['amount'].min())) * 100

    # Create the chart
    import matplotlib.pyplot as final_plt
    import matplotlib.dates as mdates
    final_fig, final_ax = final_plt.subplots(figsize=(12, 6))

    final_ax.plot(final_monthly_avg['date'], final_sales_norm, 'b-', label='Actual Sales', linewidth=3, marker='o', markersize=4)
    final_ax.plot(pred_monthly_avg['date'], final_pred_norm, 'g--', label='Seasonal Forecast Model', linewidth=3, marker='^', markersize=4)

    final_ax.set_xlabel('Date', fontsize=12)
    final_ax.set_ylabel('Normalized Sales (0-100)', fontsize=12)
    final_ax.set_title('Monthly Sales: Actual vs Seasonal Forecast Model\n(Predictions based on month + year trend)', fontsize=14)
    final_ax.legend(fontsize=11)
    final_ax.grid(True, alpha=0.3)

    # Format x-axis to show month and year
    final_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    final_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Show every 2 months
    final_plt.xticks(rotation=45)
    final_plt.tight_layout()
    final_plt.show()
    return


if __name__ == "__main__":
    app.run()
