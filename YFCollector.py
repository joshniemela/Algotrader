"""
Short program to get data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

starttime = 1645657200  # first ticker in seconds
endtime = 1653862038  # last ticker in seconds
week = 604800  # seconds in a week
symbol = "IBM"

# generate 1 week intervals from start to end
intervals = np.arange(starttime, endtime, week)

#iterate through intervals and get every 1 week interval of 1 min data from YF
frames = []
for i, interval in enumerate(intervals):
    data = yf.download(tickers=symbol, interval="2m", 
        start=datetime.fromtimestamp(interval).strftime("%Y-%m-%d"), 
        end=datetime.fromtimestamp(interval+week).strftime("%Y-%m-%d"))
    print(data)
    frames.append(data)
    print(f"{i} of {len(intervals)} downloaded")


# merge dataframes and save
df = pd.concat(frames)
df.to_csv(f"{symbol}1min.csv")


data = yf.download(tickers=symbol, period="1m", start=datetime.fromtimestamp(interval).strftime("%Y-%m-%d"), end=datetime.fromtimestamp(interval+week).strftime("%Y-%m-%d"))