# Stock Market Analysis and Screening Tool

This project provides tools to analyze stock market data and screen for potential investment opportunities.
This project was developed as part of a collaborative academic project.
My contributions included data preprocessing, trend analysis, and visualization.

## Features

1. **Stock Trend Analysis**: Visualize stock price movements over time
2. **Moving Average Analysis**: Compare short-term and long-term trends
3. **Stock Comparison**: Compare multiple stocks on the same chart
4. **Stock Screening**: Rank stocks based on performance metrics
5. **Investment Recommendations**: Get buy/sell/hold recommendations

## Files

- `stock_market_anaysis.py`: Main analysis script with visualization
- `stock_screener.py`: Stock screening and ranking tool
- `stock_recommendations.csv`: Generated stock recommendations
- `Stock-data/`: Directory containing stock data CSV files

## How to Use

### Stock Screener (Quick Analysis)
```bash
python stock_screener.py
```

This will:
1. Process a sample of stocks from the Stock-data directory
2. Calculate performance scores for each stock
3. Rank stocks based on their scores
4. Provide buy/sell recommendations
5. Save results to `stock_recommendations.csv`

### Detailed Analysis
```bash
python stock_market_anaysis.py
```

This will:
1. Analyze a single stock in detail
2. Show price charts with moving averages
3. Compare multiple stocks
4. Screen stocks and provide recommendations

## Recommendation Guide

- **STRONG BUY**: High potential for growth (score ≥ 0.5)
- **BUY**: Good investment opportunity (score ≥ 0.2)
- **HOLD**: Maintain current position (score ≥ -0.1)
- **SELL**: Consider selling (score ≥ -0.3)
- **STRONG SELL**: Strongly consider selling (score < -0.3)

## Performance Metrics

The screener evaluates stocks based on:
1. **Total Return** (40% weight): Overall price appreciation
2. **Risk-Adjusted Return** (30% weight): Sharpe ratio
3. **Trend Direction** (20% weight): Moving average analysis
4. **Volume Trend** (10% weight): Trading volume changes

## Requirements

- Python 3.x
- pandas
- matplotlib
- numpy
