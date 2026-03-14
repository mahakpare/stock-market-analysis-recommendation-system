import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------
# Step 1: Set Folder Path
# -----------------------------------
folder_path = "Stock-data"  # folder where all CSV files are stored
files = os.listdir(folder_path)

print("Total CSV files found:", len(files))
print("Example files:", files[:5])  # show first 5 file names

# -----------------------------------
# Step 2: Read One File (e.g. RELIANCE)
# -----------------------------------
# Pick a sample file that exists in the folder
sample_file = files[0] if files else "RELIANCE.NS.csv"
file_path = os.path.join(folder_path, sample_file)
print(f"Reading file: {sample_file}")
df = pd.read_csv(file_path)

print("Data shape:", df.shape)
print(df.head())

# -----------------------------------
# Step 3: Clean and Prepare Data
# -----------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Plot Closing Price
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], color='blue')
plt.title(f"Stock Trend of {sample_file.replace('.csv', '')} (2000–2020)")
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.show()

# -----------------------------------
# Step 4: Add Simple Moving Averages
# -----------------------------------
df['SMA_30'] = df['Close'].rolling(30).mean()
df['SMA_100'] = df['Close'].rolling(100).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], label='Close', color='blue')
plt.plot(df['Date'], df['SMA_30'], label='30-Day SMA', color='orange')
plt.plot(df['Date'], df['SMA_100'], label='100-Day SMA', color='red')
plt.legend()
plt.title(f"{sample_file.replace('.csv', '')} - Price vs Moving Averages")
plt.show()

# -----------------------------------
# Step 5: (Optional) Compare Multiple Companies
# -----------------------------------
# Example: Compare first two CSV files in the folder

files_to_compare = files[:2] if len(files) >= 2 else files
plt.figure(figsize=(10,5))

for file in files_to_compare:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    # Fix the FutureWarning by explicitly setting fill_method=None
    df['Return'] = df['Close'].pct_change(fill_method=None)
    plt.plot(df['Date'], df['Close'], label=file.replace('.csv', ''))

plt.title(f"Comparison of {len(files_to_compare)} Stocks (2000–2020)")
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.legend()
plt.show()

# -----------------------------------
# NEW: Step 6: Stock Performance Scoring System
# -----------------------------------
def calculate_performance_score(file_name):
    """
    Calculate a performance score for a stock based on various metrics
    """
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Handle cases where there's insufficient data
    if len(df) < 2:
        return {
            'file_name': file_name,
            'total_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'uptrend': False,
            'volume_trend': 0,
            'score': 0,
            'accuracy_metrics': {'mae': 0, 'rmse': 0, 'prediction_accuracy': 0}
        }
    
    # Fix the FutureWarning by explicitly setting fill_method=None
    df['Return'] = df['Close'].pct_change(fill_method=None)
    
    # Remove any NaN values that might have been created
    df = df.dropna(subset=['Return'])
    
    # Calculate volatility (standard deviation of returns)
    volatility = df['Return'].std() * np.sqrt(252) if df['Return'].std() > 0 else 0  # Annualized volatility
    
    # Calculate overall return
    total_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] if df['Close'].iloc[0] > 0 else 0
    
    # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
    sharpe_ratio = total_return / volatility if volatility != 0 else 0
    
    # Calculate moving average indicators (if enough data)
    if len(df) >= 200:
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        # Check if stock is in uptrend (50-day MA above 200-day MA)
        uptrend = df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] if not np.isnan(df['SMA_50'].iloc[-1]) and not np.isnan(df['SMA_200'].iloc[-1]) else False
    else:
        uptrend = False
    
    # Volume trend (if volume data exists)
    volume_trend = 0
    if 'Volume' in df.columns and len(df) > 30:
        df['Volume_SMA'] = df['Volume'].rolling(30).mean()
        volume_trend = df['Volume'].iloc[-1] / df['Volume_SMA'].iloc[-1] if df['Volume_SMA'].iloc[-1] != 0 else 1
    
    # Create a composite score
    score = (
        total_return * 0.4 +  # 40% weight to total return
        sharpe_ratio * 0.3 +  # 30% weight to risk-adjusted return
        (1 if uptrend else 0) * 0.2 +  # 20% weight to uptrend
        min(volume_trend, 2) * 0.1 if volume_trend > 0 else 0  # 10% weight to volume trend (capped at 2x)
    )
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(df)
    
    return {
        'file_name': file_name,
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'uptrend': uptrend,
        'volume_trend': volume_trend,
        'score': score,
        'accuracy_metrics': accuracy_metrics
    }

def calculate_accuracy_metrics(df):
    """
    Calculate accuracy metrics for stock price predictions
    """
    try:
        # Create a simple moving average prediction model
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['Prediction'] = df['SMA_10'].shift(1)  # Predict next day's price using previous SMA
        
        # Remove NaN values
        df_clean = df.dropna(subset=['Close', 'Prediction'])
        
        if len(df_clean) < 2:
            return {'mae': 0, 'rmse': 0, 'prediction_accuracy': 0}
        
        # Calculate MAE and RMSE
        mae = mean_absolute_error(df_clean['Close'], df_clean['Prediction'])
        rmse = np.sqrt(mean_squared_error(df_clean['Close'], df_clean['Prediction']))
        
        # Calculate directional accuracy (percentage of correct up/down predictions)
        df_clean['Actual_Change'] = df_clean['Close'].diff()
        df_clean['Predicted_Change'] = df_clean['Prediction'] - df_clean['Close'].shift(1)
        df_clean['Correct_Direction'] = (df_clean['Actual_Change'] * df_clean['Predicted_Change']) > 0
        prediction_accuracy = df_clean['Correct_Direction'].mean() * 100 if len(df_clean) > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'prediction_accuracy': prediction_accuracy
        }
    except Exception as e:
        print(f"Error calculating accuracy metrics: {str(e)}")
        return {'mae': 0, 'rmse': 0, 'prediction_accuracy': 0}

# -----------------------------------
# NEW: Step 7: Stock Recommendation System
# -----------------------------------
def get_stock_recommendation(score_data):
    """
    Provide a buy/sell recommendation based on the performance score
    """
    score = score_data['score']
    if score >= 0.5:
        return "STRONG BUY"
    elif score >= 0.2:
        return "BUY"
    elif score >= -0.1:
        return "HOLD"
    elif score >= -0.3:
        return "SELL"
    else:
        return "STRONG SELL"

# -----------------------------------
# NEW: Step 8: Analyze Top Stocks
# -----------------------------------
print("\nAnalyzing top performing stocks...")

# Analyze a sample of stocks (first 20 for demonstration to avoid long processing time)
sample_files = files[:20] if len(files) >= 20 else files
stock_scores = []

for file in sample_files:
    try:
        score_data = calculate_performance_score(file)
        stock_scores.append(score_data)
        print(f"Processed: {file}")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

# Sort stocks by performance score
stock_scores.sort(key=lambda x: x['score'], reverse=True)

# Display top 5 stocks with recommendations and accuracy metrics
print("\nTop 5 Stock Recommendations with Accuracy Metrics:")
print("-" * 120)
print(f"{'Stock':<20} {'Score':<10} {'Return':<10} {'Volatility':<12} {'Accuracy':<10} {'MAE':<12} {'RMSE':<12} {'Recommendation':<15}")
print("-" * 120)

for i, stock in enumerate(stock_scores[:5]):
    recommendation = get_stock_recommendation(stock)
    accuracy = stock['accuracy_metrics']['prediction_accuracy']
    mae = stock['accuracy_metrics']['mae']
    rmse = stock['accuracy_metrics']['rmse']
    print(f"{stock['file_name']:<20} {stock['score']:<10.2f} {stock['total_return']:<10.2f} "
          f"{stock['volatility']:<12.2f} {accuracy:<10.1f}% {mae:<12.2f} {rmse:<12.2f} {recommendation:<15}")

# -----------------------------------
# NEW: Step 9: Visualize Stock Recommendations
# -----------------------------------
# Create a bar chart of top 5 stock scores
top_5_stocks = stock_scores[:5]
stock_names = [stock['file_name'].replace('.csv', '') for stock in top_5_stocks]
scores = [stock['score'] for stock in top_5_stocks]
recommendations = [get_stock_recommendation(stock) for stock in top_5_stocks]
accuracies = [stock['accuracy_metrics']['prediction_accuracy'] for stock in top_5_stocks]

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(stock_names)), scores, color=['green' if rec in ['STRONG BUY', 'BUY'] else 
                                                       'orange' if rec == 'HOLD' else 'red' 
                                                       for rec in recommendations])
plt.xlabel('Stocks')
plt.ylabel('Performance Score')
plt.title('Top 5 Stock Performance Scores')
plt.xticks(range(len(stock_names)), stock_names, rotation=45, ha='right')

# Add value labels on bars
for i, (bar, score, acc) in enumerate(zip(bars, scores, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.2f}\n{acc:.1f}%\n{recommendations[i]}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# -----------------------------------
# NEW: Step 10: Save Recommendations to File
# -----------------------------------
# Save all stock scores and recommendations to a CSV file
recommendations_df = pd.DataFrame(stock_scores)
recommendations_df['recommendation'] = recommendations_df.apply(get_stock_recommendation, axis=1)
recommendations_df['accuracy_pct'] = recommendations_df['accuracy_metrics'].apply(lambda x: x['prediction_accuracy'])
recommendations_df['mae'] = recommendations_df['accuracy_metrics'].apply(lambda x: x['mae'])
recommendations_df['rmse'] = recommendations_df['accuracy_metrics'].apply(lambda x: x['rmse'])

# Save to CSV
output_file = "stock_recommendations.csv"
recommendations_df.to_csv(output_file, index=False)
print(f"\nStock recommendations saved to {output_file}")

print("\nAnalysis complete! The system has evaluated stocks and provided recommendations.")
print("Use the performance scores and recommendations to make informed investment decisions.")