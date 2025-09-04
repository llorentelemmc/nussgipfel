import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    return pd,


@app.cell
def __(pd):
    # Import weather data
    weather_data = pd.read_csv("data/climate/nbcn-daily_BER_previous.csv")
    
    # Import nussgipfel sales data
    sales_data = pd.read_csv("data/sales/nussgipfel.csv")
    
    return sales_data, weather_data


if __name__ == "__main__":
    app.run()
