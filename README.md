# Equity Market Trajectory Forecasting

## Overview

This project focuses on predicting the uptrend and downtrend of a company's stock using technical analysis techniques. The dataset, collected from Yahoo Finance through web scraping using stock tickers (stock symbols), contains the Highest, Lowest, Opening, and Closing values of a stock for each day.

## Methodology

Technical analysts often rely on conventional methods to predict stock trends. In this project, we initially used the 100-day Moving Average and 200-day Moving Average to predict the trend in the closing price of a stock. According to technical analysis, if the 100-day Moving Average crosses the 200-day Moving Average, it indicates an uptrend; otherwise, a downtrend is predicted.

### Data Preprocessing

Before feeding the data into the LSTM model, we performed MinMax scaling on the closing price column. This scaling brought the data within the range of 0 and 1, reducing variance and mitigating overfitting issues that may arise from high variance in the data.

### Calculations

We calculated the 100-day and 200-day moving averages for the closing prices. The predictive process began with taking the first 100 values from the training dataset and using them to predict the 101st day's closing price. This predicted value was then added to the dataset and used to predict the 102nd day's moving average, and so on. This way, we predicted values for consecutive days using previous 100-day moving averages.

## Model Upgrade to GRU

In a significant model upgrade, we transitioned from using LSTM to GRU (Gated Recurrent Unit). GRU is a type of recurrent neural network architecture that is known for its efficiency and ability to capture long-range dependencies in data, making it suitable for sequential data analysis. This upgrade reduced the training time by approximately 100 seconds, from 506 seconds to 406 seconds, primarily due to a reduction in the number of learnable parameters.

### Model Performance Improvement

The introduction of the GRU model resulted in improved performance metrics:

- **Mean Squared Error (MSE):** Reduced by 0.0096
- **R2 Score:** Improved by 0.2897

Additionally, the GRU model's weight size is 1652 KB, while the LSTM model's weight size is 2167 KB, further enhancing efficiency and resource utilization.

### Visualization and Evaluation

After predicting `Y_predicted` values using the pre-trained GRU model, we upscaled these values to their original scale using the scaling factor. By comparing `Y_test` (ground truth) with `Y_predicted`, we could visualize and quantify the offset. Plotting the two helped us assess the model's performance.

## How GRU Works

In the traditional Vanilla RNN, the network struggles to capture long-range dependencies due to vanishing and exploding gradient problems. GRU addresses these issues by incorporating memory cells and gates that regulate the flow of information, enabling it to capture long-term dependencies efficiently.

## Getting Started

To replicate or extend this work, follow these steps:

1. Collect stock data from Yahoo Finance using stock tickers.
2. Perform MinMax scaling on the closing price column.
3. Calculate the 100-day and 200-day moving averages.
4. Split the data into training and testing sets (70%-30%).
5. Train the GRU model using the training dataset.
6. Predict the trend using the trained model and assess its performance.

Feel free to modify the code and experiment with different hyperparameters to achieve even better results.

## Dependencies

- Python
- Pandas
- Numpy
- Tensorflow
- Matplotlib
