import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt


class ImpliedVolatilityCalculator:
    def __init__(self, ticker, max_iterations=5):
        """
        Initialize the calculator with a specific stock ticker
        
        :param ticker: Stock ticker symbol (e.g., 'AAPL')
        :param max_iterations: Maximum number of attempts to calculate IV
        """
        self.ticker = ticker
        self.stock_data = None
        self.options_data = None
        self.max_iterations = max_iterations
    
    def fetch_stock_data(self):
        """
        Fetch current stock price and risk-free rate
        """
        stock = yf.Ticker(self.ticker)
        
        # Get current stock price
        self.current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # Fetch options chain
        try:
            # Get all available expiration dates
            self.options_data = stock.options
            self.options_chain = self._process_options_chain(stock)
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None
        
        # Use 10-year Treasury rate as risk-free rate (approximation)
        self.risk_free_rate = yf.Ticker('^TNX').history(period="1d")['Close'].iloc[-1] / 100
        
        return self.current_price
    
    def _process_options_chain(self, stock):
        """
        Process options chain for calls and puts with more comprehensive data gathering
        
        :param stock: yfinance Ticker object
        :return: DataFrame with options data
        """
        options_data = []
        
        # Iterate through all available expiration dates
        for date in self.options_data:
            try:
                # Fetch both calls and puts
                calls = stock.option_chain(date).calls
                puts = stock.option_chain(date).puts
                
                # Add expiration date and process
                for df in [calls, puts]:
                    df['Expiration'] = date
                    options_data.append(df)
            except Exception as e:
                print(f"Error processing options for {date}: {e}")
        
        # Combine and filter out low-volume or invalid options
        if options_data:
            combined_options = pd.concat(options_data)
            # Filter out options with very low volume or open interest
            filtered_options = combined_options[
                (combined_options['volume'] > 10) & 
                (combined_options['openInterest'] > 5)
            ]
            return filtered_options
        
        return pd.DataFrame()
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes Call Option Price
        
        :param S: Current stock price
        :param K: Strike price
        :param T: Time to expiration (in years)
        :param r: Risk-free rate
        :param sigma: Volatility
        :return: Theoretical call option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def implied_volatility(self, market_price, S, K, T, r, option_type='call'):
        """
        Calculate Implied Volatility using Brent's method with multiple attempts
        
        :param market_price: Current market price of the option
        :param S: Current stock price
        :param K: Strike price
        :param T: Time to expiration (in years)
        :param r: Risk-free rate
        :param option_type: 'call' or 'put'
        :return: Implied volatility
        """
        def option_price_diff(sigma):
            """
            Calculate the difference between market and theoretical option price
            """
            if option_type == 'call':
                theo_price = self.black_scholes_call(S, K, T, r, sigma)
            else:
                # Put-call parity for put option
                theo_price = (self.black_scholes_call(S, K, T, r, sigma) 
                              - S + K * np.exp(-r * T))
            
            return theo_price - market_price
        
        # Try multiple volatility estimation ranges
        volatility_ranges = [
            (0.0001, 5),    # Conservative range
            (0.0001, 10),   # Wider range
            (0.0001, 20),   # Very wide range
            (0.01, 5),      # Adjusted lower bound
            (0.1, 2)        # Tighter, mid-range estimate
        ]
        
        for lower, upper in volatility_ranges:
            try:
                implied_vol = brentq(option_price_diff, lower, upper)
                return implied_vol
            except ValueError:
                continue
        
        return None
    
    def calculate_implied_volatilities(self):
        """
        Calculate implied volatilities for options in the chain
        
        :return: DataFrame with implied volatilities
        """
        if self.stock_data is None:
            self.fetch_stock_data()
        
        results = []
        
        for _, option in self.options_chain.iterrows():
            try:
                # Convert expiration to years
                expiration_date = pd.to_datetime(option['Expiration'])
                time_to_expiry = (expiration_date - pd.Timestamp.now()).days / 365.25
                
                # Only process options with reasonable time to expiry
                if 0 < time_to_expiry <= 1:
                    iv = self.implied_volatility(
                        market_price=option['lastPrice'],
                        S=self.current_price,
                        K=option['strike'],
                        T=time_to_expiry,
                        r=self.risk_free_rate,
                        option_type='call' if option['contractSymbol'].startswith('C') else 'put'
                    )
                    
                    # Only add if implied volatility is calculated
                    if iv is not None and not np.isnan(iv):
                        results.append({
                            'Ticker': self.ticker,
                            'Current Stock Price': self.current_price,
                            'Strike': option['strike'],
                            'Expiration': option['Expiration'],
                            'Option Type': 'Call' if option['contractSymbol'].startswith('C') else 'Put',
                            'Market Price': option['lastPrice'],
                            'Volume': option['volume'],
                            'Open Interest': option['openInterest'],
                            'Implied Volatility': iv,
                            'Time to Expiry (Years)': time_to_expiry,
                            'Risk-Free Rate': self.risk_free_rate
                        })
            except Exception as e:
                print(f"Error calculating IV for option: {e}")
        
        # Convert to DataFrame and return
        return pd.DataFrame(results)
    
    def export_to_excel(self, results):
        """
        Export results to an Excel file
        
        :param results: DataFrame with implied volatility results
        """
        # Create output directory if it doesn't exist
        output_dir = 'implied_volatility_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{output_dir}/{self.ticker}_implied_volatility_{timestamp}.xlsx'
        
        # Export to Excel
        results.to_excel(filename, index=False)
        print(f"Results exported to {filename}")
        return filename

class ImpliedVolatilityValidator:
    def __init__(self, ticker):
        """
        Initialize the validator with a specific stock ticker
        
        :param ticker: Stock ticker symbol (e.g., 'AAPL')
        """
        self.ticker = ticker
        self.stock_data = None
        self.options_data = None
    
    def fetch_historical_prices(self, start_date=None, end_date=None):
        """
        Fetch historical stock prices
        
        :param start_date: Start date for historical data (default: 1 year ago)
        :param end_date: End date for historical data (default: today)
        :return: DataFrame with historical stock prices
        """
        # If no dates provided, use 1 year of historical data
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        stock = yf.Ticker(self.ticker)
        hist_data = stock.history(start=start_date, end=end_date)
        
        return hist_data
    
    def calculate_realized_volatility(self, prices=None, window=30):
        """
        Calculate realized volatility using log returns
        
        :param prices: DataFrame with historical prices (will fetch if None)
        :param window: Rolling window for volatility calculation
        :return: Series of realized volatilities
        """
        if prices is None:
            prices = self.fetch_historical_prices()
        
        # Calculate log returns
        log_returns = np.log(prices['Close'] / prices['Close'].shift(1))
        
        # Calculate realized volatility (annualized)
        realized_vol = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualize
        
        return realized_vol
    
    def compare_volatilities(self, implied_volatilities=None, historical_prices=None):
        """
        Compare implied volatility with realized volatility
        
        :param implied_volatilities: DataFrame with implied volatilities
        :param historical_prices: DataFrame with historical stock prices
        :return: DataFrame comparing volatilities
        """
        # Fetch historical prices if not provided
        if historical_prices is None:
            historical_prices = self.fetch_historical_prices()
        
        # Calculate realized volatility
        realized_vol = self.calculate_realized_volatility(historical_prices)
        
        # If implied volatilities not provided, calculate them
        if implied_volatilities is None:
            calculator = ImpliedVolatilityCalculator(self.ticker)
            calculator.fetch_stock_data()
            implied_volatilities = calculator.calculate_implied_volatilities()
        
        # Prepare comparison DataFrame
        comparison = pd.DataFrame({
            'Date': historical_prices.index,
            'Realized Volatility': realized_vol
        })
        
        # Merge with implied volatilities
        if not implied_volatilities.empty:
            # Average implied volatility for each expiration date
            implied_by_date = implied_volatilities.groupby('Expiration')['Implied Volatility'].mean()
            
            # Add implied volatility to comparison
            for expiration, iv in implied_by_date.items():
                comparison[f'Implied Vol ({expiration})'] = iv
        
        return comparison
    
    def visualize_volatility_comparison(self, comparison_data):
        """
        Create visualization of volatility comparison
        
        :param comparison_data: DataFrame with volatility comparison
        """
        plt.figure(figsize=(12, 6))
        
        # Plot realized volatility
        plt.plot(comparison_data.index, 
                 comparison_data['Realized Volatility'], 
                 label='Realized Volatility', 
                 color='blue')
        
        # Plot implied volatilities
        implied_cols = [col for col in comparison_data.columns if 'Implied Vol' in col]
        for col in implied_cols:
            plt.axhline(y=comparison_data[col].iloc[0], 
                        color='red', 
                        linestyle='--', 
                        label=col)
        
        plt.title(f'{self.ticker} - Volatility Comparison')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        
        # Save the plot
        output_dir = 'volatility_output'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/{self.ticker}_volatility_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    def accuracy_metrics(self, comparison_data):
        """
        Calculate accuracy metrics for implied volatility
        
        :param comparison_data: DataFrame with volatility comparison
        :return: Dictionary of accuracy metrics
        """
        # Prepare results dictionary
        metrics = {}
        
        # Identify implied volatility columns
        implied_cols = [col for col in comparison_data.columns if 'Implied Vol' in col]
        
        # Remove NaN values
        clean_data = comparison_data.dropna(subset=['Realized Volatility'] + implied_cols)
        
        for col in implied_cols:
            # Mean Absolute Error
            mae = np.abs(clean_data['Realized Volatility'] - clean_data[col]).mean()
            
            # Mean Squared Error
            mse = np.mean((clean_data['Realized Volatility'] - clean_data[col])**2)
            
            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            
            # Correlation
            correlation = clean_data['Realized Volatility'].corr(clean_data[col])
            
            metrics[col] = {
                'Mean Absolute Error': mae,
                'Root Mean Squared Error': rmse,
                'Correlation': correlation
            }
        
        return metrics

