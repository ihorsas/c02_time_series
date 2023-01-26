import statsmodels.tsa.api as smt
from statsmodels.datasets import get_rdataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.stats import diagnostic as diag
import numpy as np

# exporting the dataset for monthly average temperature in New York City from January 1959 to December 1997
co2 = get_rdataset("co2")
co2_data = co2.data["value"]

# function to fit and evaluate ARIMA models
def fit_arima(data, p, d, q):
    model = smt.ARIMA(data, order=(p, d, q))
    results = model.fit()
    print(f"AIC: {results.aic}")
    print(f"BIC: {results.bic}")
    residuals = results.resid
    # visualizing acf and pacf
    smt.graphics.plot_acf(residuals, lags=40)
    smt.graphics.plot_pacf(residuals, lags=40)
    plt.show()

    # divide data into train and test set
    test_size = int(len(data) * 0.2)
    train_data, test_data = data[:-test_size], data[-test_size:]

    # tit model using train data
    model = smt.ARIMA(train_data, order=(p, d, q))
    results = model.fit()

    # verify the appropriateness by checking if the residuals are white noise
    p_value = diag.acorr_ljungbox(residuals, lags=[20], return_df=False)
    print(f"p_value: {p_value} ")
    if p_value[0] > 0.05:
        print("The residuals are likely white noise.")
    else:
        print("The residuals are not white noise.")

    # make predictions using test data
    predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    print(f"RMSE: {rmse}")

    # plot actual vs predicted values
    plt.plot(test_data, label='Actual')
    plt.plot(predictions, label='Predicted', color='r')
    plt.legend()
    plt.show()
    return rmse, results


print("Co2 Data:")

print("Fitting arima for p=10, d=2, q=1")
# model 1 will be the best based on experiments
rmse_1, model_1 = fit_arima(co2_data, 10, 2, 1)
#
print("Fitting arima for p=1, d=1, q=1")
rmse_2, model_2 = fit_arima(co2_data, 1, 2, 10)

print("Fitting arima for p=28, d=1, q=10")
rmse_3, model_3 = fit_arima(co2_data, 10, 2, 10)

models = {
    rmse_1: model_1,
    rmse_2: model_2,
    rmse_3: model_3
}

# choose the model with the lowest RMSE
minimum_rmse = min(models.keys())
print(f"Minimum rmse {minimum_rmse}")
best_model = models[minimum_rmse]

# use the best model to predict future values for the next year
future_predictions = best_model.predict(start=len(co2_data), end=len(co2_data) + 12)

# plot actual vs predicted values
plt.plot(co2_data, label='Actual', color='b')
plt.plot(future_predictions, label='Future Predicted', color='r')
plt.legend()
plt.show()
