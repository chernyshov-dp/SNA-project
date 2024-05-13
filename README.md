# Predicting Stock Prices using GCN and LSTM Networks

## Key Idea of the Project

The goal of this project is to develop an accurate predictive model for stock prices using a combination of Graph Convolutional Neural Networks (GCN) and Long Short-Term Memory (LSTM) networks.

Predicting stock prices accurately is of paramount importance for investors and traders to make informed decisions and manage risks effectively. Traditional methods often rely on technical analysis and historical data. However, by leveraging the power of neural networks and graph analysis, we aim to create a more robust and accurate predictive model.

Similar approaches combining graph analysis and neural networks have been used in various domains, including social network analysis, recommendation systems, and fraud detection. In recent years, the application of such techniques to financial data analysis, including stock price prediction, has gained attention.

To achieve our goal, we plan to follow these steps:

1. Data Collection;
2. Graph Construction;
3. Model Development.

## Dataset Required to Do the Project

Historical stock price data will be collected from reputable financial data sources such as Yahoo Finance, Alpha Vantage, etc. The dataset should include features such as opening price, closing price, highest price, lowest price, and trading volume for a specific time period. Data should cover a sufficiently long time frame to capture various market conditions and trends. Data quality is essential for reliable predictions. Therefore, we will ensure that the data is accurate and free from errors.

## Anticipated SNA Methods

- **Graph Construction**: Build a graph representation of the stock price data, where each node represents a price change, and edges connect consecutive price changes. This approach allows us to capture the sequential relationship between different price changes and represent the data in a structured format suitable for graph analysis.
    
- **Graph Convolutional Neural Network (GCN)**: Train a GCN to analyze the graph structure of the stock price data, capturing the relationships between different price changes. GCNs have been proven effective in analyzing graph-structured data and capturing complex relationships between nodes in the graph.
    
- **Long Short-Term Memory (LSTM) Network**: Pass the output of the GCN to an LSTM network to capture temporal dependencies in the data and make predictions based on historical trends. LSTMs are well-suited for capturing temporal dependencies in sequential data, making them ideal for analyzing time series data like stock prices.
    
- **Model Evaluation**: Evaluate the performance of the combined GCN-LSTM model using appropriate metrics.
