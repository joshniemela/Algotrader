"""
Program to test the effectiveness of labelling based on triple-barrier windows, algorithm used is a random forest classifier.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle


def triplebarrier(data, deviations, spread):
    startprice = data[0]
    std = np.std(data)
    upper = startprice * (1 + spread) + std * deviations
    lower = startprice * (1 / (1 + spread)) - std * deviations
    for price in data:
        if price > upper:
            return 1  # Buy
        elif price < lower:
            return -1  # Sell
    return 0  # Hold


"""
Function to skip preprocessing on any data containing NaN values
"""
def scaleunclean(data):
    if np.any(np.isnan(data)):
        return data
    else:
        return preprocessing.scale(data)


PROGRAM_PATH = os.getcwd()

filename = f"{PROGRAM_PATH}/SPY1day.csv"
df = pd.read_csv(filename)
close = df.Close.values

# label parameters
window_length = 30  # How many candles to window into the future
deviations = 2  # ± Standard deviations of start price
ts_length = 25  # Length of time series used for training data
test_val_split = 0.5
spread = 0.02  # Expected spread in percent

labels = np.empty(len(close)) * np.nan
for i in range(len(close) - window_length):
    labels[i] = triplebarrier(close[i : i + window_length], deviations, spread)

samples = np.empty(shape=(len(close), ts_length)) * np.nan
for i in range(len(close) - ts_length):
    samples[i] = close[i : i + ts_length]

# Process the samples
samples = np.apply_along_axis(scaleunclean, 1, samples)
# remove  nans
samples = samples[window_length:-window_length]
labels = labels[window_length:-window_length]

X_train, X_test, y_train, y_test = train_test_split(
    samples, labels, test_size=test_val_split
)

X_train, y_train = shuffle(X_train, y_train)

model = RandomForestClassifier(n_estimators=750)
model.fit(X_train, y_train)

guessed_labels = model.predict(X_test)
cm = confusion_matrix(y_test, guessed_labels, normalize="true", labels=model.classes_)


fig, (ax1, ax2) = plt.subplots(2)
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Sell", "Hold", "Buy"]
).plot(ax=ax1, cmap="Greys")

print(model.feature_importances_)
# Trade bot thing
account = [close[len(X_train) + window_length]]
isHolding = False

for i, current_price in enumerate(close[len(X_train) + window_length : -window_length]):
    if guessed_labels[i] == 1:
        if not isHolding:
            bought_at = current_price
            isHolding = True
        else:
            account.append(account[-1])
    elif guessed_labels[i] == -1:
        if isHolding:
            account.append(account[-1] + current_price - bought_at)
            isHolding = False
        else:
            account.append(account[-1])
    else:
        account.append(account[-1])

ax2.plot(close[len(X_train) + window_length : -window_length], label="Price")
ax2.plot(account, label="Trader according to predicted signals")

# trade bot thing
account = [close[len(X_train) + window_length]]
isHolding = False

for i, current_price in enumerate(
    close[len(X_train) + window_length : -window_length - 1]
):
    if y_train[i] == 1:
        if not isHolding:
            bought_at = current_price
            isHolding = True
        else:
            account.append(account[-1])
    elif y_train[i] == -1:
        if isHolding:
            account.appentslen(account[-1] + current_price - bought_at)
            isHolding = False
        else:
            account.append(account[-1])
    else:
        account.append(account[-1])
ax2.plot(account, label="Trader according to label-dataset")

ax2.legend()
ax2.set_yscale("log")
plt.show()
