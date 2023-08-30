# Equity_Market_Trajectory_Forecasting

This project focuses on predicting the uptrend and downtrend of a company's stock using technical analysis techniques. The dataset, collected from Yahoo Finance through web scraping using stock tickers (stock symbols), contains the Highest, Lowest, Opening, and Closing values of a stock for each day.

## Methodology

Technical analysts often rely on conventional methods to predict stock trends. In this project, we use the 100-day Moving Average and 200-day Moving Average to predict the trend in the closing price of a stock. According to technical analysis, if the 100-day Moving Average crosses the 200-day Moving Average, it indicates an uptrend; otherwise, a downtrend is predicted.

### Data Preprocessing

Before feeding the data into the LSTM model, we perform MinMax scaling on the closing price column. This scaling brings the data within the range of 0 and 1, reducing variance and mitigating overfitting issues that may arise from high variance in the data.

### Calculations

We calculate the 100-day and 200-day moving averages for the closing prices. The predictive process begins with taking the first 100 values from the training dataset and using them to predict the 101st day's closing price. This predicted value is then added to the dataset and used to predict the 102nd day's moving average, and so on. This way, we predict values for consecutive days using previous 100-day moving averages.

### Training and Testing

For training, we use the first 100 days' values (100 columns) as input (`X_train`), and the corresponding 101st day's value is our ground truth (`Y_train`). This pattern is repeated for all instances in the training set, utilizing LSTM models. We use Mean Squared Error as the loss function and the Adam optimizer for training.

Similarly, `X_test` and `Y_test` are structured for testing purposes. To predict the first value in the test dataset, we use the last 100 days' data from the training dataset.

### Visualization and Evaluation

After predicting `Y_predicted` values using the pre-trained LSTM model, we upscale these values to their original scale using the scaling factor. By comparing `Y_test` (ground truth) with `Y_predicted`, we can visualize and quantify the offset. Plotting the two helps us assess the model's performance.

## Evaluation Metrics

- **Mean Squared Error (MSE):** 8.18
- **R2 Score:** 0.89

## How LSTM Works

In the traditional Vanilla RNN, the network struggles to capture long-range dependencies due to vanishing and exploding gradient problems. LSTM addresses these issues by incorporating memory cells and gates that regulate the flow of information, enabling it to capture long-term dependencies and making it suitable for sequential data analysis.

## Getting Started

To replicate or extend this work, follow these steps:

1. Collect stock data from Yahoo Finance using stock tickers.
2. Perform MinMax scaling on the closing price column.
3. Calculate the 100-day and 200-day moving averages.
4. Split the data into training and testing sets (70%-30%).
5. Train the LSTM model using the training dataset.
6. Predict the trend using the trained model and assess its performance.

Feel free to modify the code and experiment with different hyperparameters to achieve better results.

## Dependencies

- Python
- Pandas
- Numpy
- Tensorflow
- Matplotlib
