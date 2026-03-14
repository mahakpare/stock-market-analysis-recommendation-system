import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_performance_score(file_path, file_name):
    """
    Calculate a performance score for a stock based on various metrics
    """
    try:
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
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
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

def main():
    # Set folder path
    folder_path = "Stock-data"
    
    # Get list of files
    files = os.listdir(folder_path)
    print(f"Found {len(files)} stock files")
    
    # Process only first 10 files for demo
    sample_files = files[:10]
    print(f"Processing {len(sample_files)} stocks for demonstration...")
    
    # Calculate scores for each stock
    stock_scores = []
    for file in sample_files:
        file_path = os.path.join(folder_path, file)
        score_data = calculate_performance_score(file_path, file)
        stock_scores.append(score_data)
        print(f"Processed: {file}")
    
    # Sort by score
    stock_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Display results with accuracy metrics
    print("\n" + "="*120)
    print("STOCK SCREENING RESULTS")
    print("="*120)
    print(f"{'Stock':<20} {'Score':<10} {'Return':<10} {'Volatility':<12} {'Accuracy':<10} {'MAE':<12} {'RMSE':<12} {'Recommendation':<15}")
    print("-"*120)
    
    for stock in stock_scores:
        recommendation = get_stock_recommendation(stock)
        accuracy = stock['accuracy_metrics']['prediction_accuracy']
        mae = stock['accuracy_metrics']['mae']
        rmse = stock['accuracy_metrics']['rmse']
        print(f"{stock['file_name']:<20} {stock['score']:<10.2f} {stock['total_return']:<10.2f} "
              f"{stock['volatility']:<12.2f} {accuracy:<10.1f}% {mae:<12.2f} {rmse:<12.2f} {recommendation:<15}")
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(stock_scores)
    df['recommendation'] = df.apply(get_stock_recommendation, axis=1)
    df['accuracy_pct'] = df['accuracy_metrics'].apply(lambda x: x['prediction_accuracy'])
    df['mae'] = df['accuracy_metrics'].apply(lambda x: x['mae'])
    df['rmse'] = df['accuracy_metrics'].apply(lambda x: x['rmse'])
    
    # Select columns for CSV
    output_df = df[['file_name', 'total_return', 'volatility', 'sharpe_ratio', 'uptrend', 'volume_trend', 'score', 'accuracy_pct', 'mae', 'rmse', 'recommendation']]
    output_df.to_csv('stock_recommendations.csv', index=False)
    print(f"\nResults saved to stock_recommendations.csv")
    
    print("\nRecommendation Guide:")
    print("- STRONG BUY: High potential for growth")
    print("- BUY: Good investment opportunity")
    print("- HOLD: Maintain current position")
    print("- SELL: Consider selling")
    print("- STRONG SELL: Strongly consider selling")

if __name__ == "__main__":
    main()