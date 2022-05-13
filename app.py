import os
import errno
import pandas as pd
import matplotlib.pylab as plt
import statsmodels
from scipy.stats import mode
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings


def create_statistics_table_num(df):
    df_n = df.copy()
    df_n = df_n.drop("Date", axis=1)
    describe = df_n.describe()
    temp = pd.DataFrame(index=['range', 'median', 'variance', 'skew', 'kurt', 'coefficient'], columns=df_n.columns)
    for col in df_n.columns:
        list_temp = [df_n[col].max() - df_n[col].min(), df_n[col].median(), df_n[col].var(), df_n[col].skew(),
                     df_n[col].kurt(), df_n[col].std() / df_n[col].mean()]
        temp[col] = list_temp
    describe = pd.concat([describe, temp])
    return describe


def create_statistics_table_cat(df):
    df_n = df.copy()
    cols = df_n.columns
    temp = pd.DataFrame(index=['Variation Ratio'], columns=cols)
    for col in cols:
        list_temp = [1 - mode(df_n[col])[1][0] / len(df_n[col])]
        temp[col] = list_temp
    return temp


def test_stationarity(ts):
    movingAVG = ts.rolling(window=12).mean()
    movingSTD = ts.rolling(window=12).std()
    plt.title = "Rolling mean and std"
    orig = plt.plot(ts, color='black', label='Original')
    mean = plt.plot(movingAVG, color='red', label='Mean')
    std = plt.plot(movingSTD, color='blue', label='Std')
    plt.legend(loc='best')
    plt.show(block=False)

    dftest = statsmodels.tsa.stattools.adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.66)
    print(X[0:train_size])
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        warnings.filterwarnings("ignore")
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE = %.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


def fetch_data(file_path):
    print("\n$$$$$ Perform basic analysis of the data ...")
    flag = os.path.isfile(file_path)
    if flag:
        print(f"The file {file_path} exists")
        return file_path
    else:
        print(f"The file {file_path} does not exist")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)


def part_one(file_path):
    print("\n$$$$$ Reading Data ...")
    data = pd.read_csv(file_path, error_bad_lines=False, parse_dates=True)
    data["Date"] = pd.to_datetime(data["Date"])

    print("\n$$$$$ Different data types? ...")
    print(data.info())

    print("\n$$$$$ Missing values? ...")
    print(data.isnull().any())
    print('\n\t\tmissing num\t\tmissing %')
    for string in data.columns:
        missing_num = data[string].isnull().sum()
        missing_percent = data[string].isnull().sum() / len(data[string])
        print(string + f'\t\t{missing_num}\t\t{missing_percent}')

    print("\n$$$$$ Shape of the dataset? ...")
    print(create_statistics_table_num(data))
    print(create_statistics_table_cat(data))

    print("\n$$$$$ Visualizing Data ...")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.grid(True)
    ax.plot(data["Date"], data["EUR/USD"], label="EUR/USD", color="r")
    ax.plot(data["Date"], data["GBP/USD"], label="GBP/USD", color="c")
    ax.plot(data["Date"], data["USD/CHF"], label="USD/CHF", color="m")
    ax.plot(data["Date"], data["USD/CAD"], label="USD/CAD", color="y")
    ax.plot(data["Date"], data["AUD/USD"], label="AUD/USD", color="k")
    ax.set_xlabel("Date")
    ax.set_ylabel("Currency")
    ax.set_title("currency_pair_prices")
    plt.legend()
    fig.show()

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('USD/JPY')
    ax.grid(True)
    ax.plot(data["Date"], data["USD/JPY"], label="USD/JPY", color="b")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/JPY")
    plt.legend()
    fig.show()

    print("\n$$$$$ Calculating correlations between the time series ...")
    print(data.diff().corr())

    print("\n$$$$$ Deal with the missing values in the time series ...")

    print("\n$$$$$ Interpolating data ...")
    data = data.interpolate()  # ffill(), bfill(), or apply limit to it

    print("\n$$$$$ Visualize new data ...")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.grid(True)
    ax.plot(data["Date"], data["EUR/USD"], label="EUR/USD", color="r")
    ax.plot(data["Date"], data["GBP/USD"], label="GBP/USD", color="c")
    ax.plot(data["Date"], data["USD/CHF"], label="USD/CHF", color="m")
    ax.plot(data["Date"], data["USD/CAD"], label="USD/CAD", color="y")
    ax.plot(data["Date"], data["AUD/USD"], label="AUD/USD", color="k")
    ax.set_xlabel("Date")
    ax.set_ylabel("Currency")
    ax.set_title("currency_pair_prices")
    plt.legend()
    fig.show()

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('USD/JPY')
    ax.grid(True)
    ax.plot(data["Date"], data["USD/JPY"], label="USD/JPY", color="b")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/JPY")
    plt.legend()
    fig.show()

    return data


def outliers_clean_up(data, keyy):
    print("\n$$$$$ Outliers? ...\n")
    data1 = data.filter(["Date", keyy], axis=1)
    data1 = data1.set_index("Date")

    Q1 = data1.quantile(0.25)
    Q3 = data1.quantile(0.75)
    IQR = Q3 - Q1
    print("Inter-quartile Range (IQR): ", IQR)
    print(data1 < (Q1 - 1.5 * IQR) | (data1 > (Q3 + 1.5 * IQR)))

    print(data1[keyy].quantile(0.05))
    lower = data1[keyy].quantile(0.05)
    print(data1[keyy].quantile(0.95))
    upper = data1[keyy].quantile(0.95)

    data1[keyy] = np.where(data1[keyy] < lower, lower, data1[keyy])
    data1[keyy] = np.where(data1[keyy] > upper, upper, data1[keyy])

    print(data1[keyy].skew())
    return data1


def log_returns(data1):
    print("\n$$$$$ Calculate the log returns USD/JPY ...")
    returns = data1.pct_change(1)
    print(returns.head())
    log_returns = np.log(data1).diff()
    print(log_returns.head())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    for c in log_returns:
        ax1.plot(log_returns.index, log_returns[c].cumsum(), label=str(c))
    ax1.set_ylabel('Cumulative log returns')
    ax1.legend(loc='best')
    for c in log_returns:
        ax2.plot(log_returns.index, 100 * (np.exp(log_returns[c].cumsum()) - 1), label=str(c))
    ax2.set_ylabel('Total relative returns (%)')
    ax2.legend(loc='best')
    plt.show()

    print("\nLast day returns\n")
    r_t = log_returns.tail(1).transpose()
    print(r_t)


def visualize_trend_seasonality_resid(data1):
    decomposition = seasonal_decompose(data1)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(data1, label='Original Data')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    decomposedLogData = residual
    decomposedLogData.dropna(inplace=True)
    print("\n >> The stationarity of decomposed Log Data")
    print(test_stationarity(decomposedLogData))


def dickey_fuller_test(data, k):
    print("\n$$$$$ Verify the stationarity: >> Dickey Fuller Test")
    print(
        "\nA test used to determine whether a unit root or a feature that can cause issues "
        "in statistical inference is present in an autoregressive model.\n")
    data1 = data.filter(["Date", k], axis=1)
    data1 = data1.set_index("Date")
    print(test_stationarity(data1))

    exponentialWeightedAVG = data1.ewm(halflife=12, min_periods=0, adjust=True).mean()
    plt.plot(exponentialWeightedAVG, color='red')
    plt.plot(data1)
    plt.show()

    print("\n >> The stationarity of LogScale minus moving exponential Weighted AVG")
    indexDatasetLogScaleMinusMovingEWA = data1 - exponentialWeightedAVG
    print(test_stationarity(indexDatasetLogScaleMinusMovingEWA))

    indexDatasetLogScaleShift = data1 - data1.shift()
    plt.plot(indexDatasetLogScaleShift)
    plt.show()
    return indexDatasetLogScaleShift


def arima_forecasting(data1, ko):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(data1, order=(4, 1, 1), seasonal_order=(0, 1, 1, 12),
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    results_ARIMA = model.fit()
    print(results_ARIMA.summary())
    results_ARIMA.plot_diagnostics(figsize=(15, 12))
    plt.show()

    pred = results_ARIMA.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
    pred_ci = pred.conf_int()

    ax = data1['2018-01-01':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2020-01-01'), data1.index[-1],
                     alpha=.1, zorder=-1)
    ax.set_xlabel('Date')
    ax.set_ylabel(ko)
    plt.legend()
    plt.show()

    # Extract the predicted and true values of our time series
    y_forecasted = pred.predicted_mean
    y_truth = data1['2020-01-01':]
    # Compute the mean square error
    mse = mean_squared_error(y_truth , y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    pred_uc = results_ARIMA.get_forecast(steps=50)
    pred_ci = pred_uc.conf_int()

    ax = data1.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel(ko)
    plt.legend()
    plt.show()


def run():
    file_path = "currency_pair_prices.csv"

    fetch_data(file_path)

    data = part_one(file_path)
    arr = ["USD/JPY", "EUR/USD", "GBP/USD", "USD/CHF", "USD/CAD", "AUD/USD"]
    for k in arr:
        print("\n Currency Pair:%s", k)
        data1 = outliers_clean_up(data, k)  # for loop? to go through all columns
        log_returns(data1)
        visualize_trend_seasonality_resid(data1)
        # logScaleShift = dickey_fuller_test(data, k)

    data1 = outliers_clean_up(data, "EUR/USD")
    # arima_forecasting(data1, "EUR/USD")

    # right parameters?
    p_values = [4, 10]
    d_values = range(1, 3)
    q_values = range(1, 3)
    evaluate_models(data1["EUR/USD"], p_values, d_values, q_values)
