import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
import sys

def Custom_EMA(data,
               source,
               period=9,
               plot=False,
               chart_type='line'  # 'line' or 'candle'
               ):
    """
    Custom_EMA: Load historical data from a local CSV file (based on the 'Data' name),
    calculate an Exponential Moving Average (EMA) for a user-defined period,
    and optionally plot the results as a line or candlestick chart.

    Parameters:
    - Data (str): Base name of the CSV file to load (e.g., 'BTCUSDT3600' refers to 'BTCUSDT3600.csv').
    - Start (str): Start date for the data range in 'YYYY-MM-DD' format.
    - End (str): End date for the data range in 'YYYY-MM-DD' format.
    - Ema_Period (int): Number of periods for the EMA calculation. Default is 9.
    - Plot (bool): If True, plots the closing price and EMA. Default is False.
    - chart_type (str): Type of plot — 'line' for a simple line chart or 'candle' for candlestick chart.

    Notes:
    - The function expects the CSV file to contain at least 'Date', 'Open', 'High', 'Low', and 'Close' columns.
    """

    # Load the CSV file and parse 'Date' column as datetime


    if data.empty:
        print("Failed to load data. Check file or contents.")
        return


    if data.empty:
        print("No data available in the selected time range.")
        return

    # Calculate EMA
    output = ta.ema(data[source], length=period)


    # Plotting
    if plot:
        if chart_type == 'line':
            data['EMA'] = output.values 
            plt.figure(figsize=(12, 6))
            plt.plot(data['close'], label='Close Price', linewidth=1.5)
            plt.plot(data['EMA'], label=f'EMA {period}', linewidth=2, linestyle='--')
            plt.title(f'Closing Price and EMA {period}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif chart_type == 'candle':
            df_candle = data.set_index('date')[['open', 'high', 'low', 'close']]
            df_candle['EMA'] = output.values
            mpf.plot(df_candle,
                     type='candle',
                     style='charles',
                     title=f'Candlestick Chart with EMA {period}',
                     ylabel='Price',
                     mav=(period,),
                     volume=False,
                     tight_layout=True)

        else:
            print(f"Unknown chart_type '{chart_type}'. Use 'line' or 'candle'.")

    return output

