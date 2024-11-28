from BSM import ImpliedVolatilityCalculator, ImpliedVolatilityValidator
import traceback

def main():
    # Example tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Aggregate results for all tickers
    all_results = []
    
    for ticker in tickers:
        try:
            print(f"\nCalculating Implied Volatility for {ticker}")
            calculator = ImpliedVolatilityCalculator(ticker)
            
            # Fetch stock and options data with full error details
            try:
                stock_data = calculator.fetch_stock_data()
                if not stock_data:
                    print(f"Failed to fetch stock data for {ticker}")
                    continue
            except Exception as e:
                print(f"Error fetching stock data for {ticker}:")
                print(traceback.format_exc())
                continue
            
            # Calculate implied volatilities
            try:
                iv_results = calculator.calculate_implied_volatilities()
                
                # Only proceed if results are not empty
                if iv_results.empty:
                    print(f"No valid options found for {ticker}")
                    continue
                
                # Add results to aggregate list
                all_results.append(iv_results)
                
                # Export individual ticker results
                try:
                    export_filename = calculator.export_to_excel(iv_results)
                    print(f"Results exported to {export_filename}")
                except Exception as export_error:
                    print(f"Error exporting results for {ticker}: {export_error}")
                
                # Print some statistics
                print(f"Calculated Implied Volatilities for {ticker}:")
                print(f"Total Options Processed: {len(iv_results)}")
                print(f"Average Implied Volatility: {iv_results['Implied Volatility'].mean():.2%}")
                print(f"Volatility Range: {iv_results['Implied Volatility'].min():.2%} - {iv_results['Implied Volatility'].max():.2%}")
            
            except Exception as iv_error:
                print(f"Error calculating implied volatilities for {ticker}:")
                print(traceback.format_exc())
                continue
        
            # Validate volatility
            print(f"\nValidating Volatility for {ticker}")
            validator = ImpliedVolatilityValidator(ticker)
            
            # Fetch historical prices
            historical_prices = validator.fetch_historical_prices()
            
            # Calculate and compare volatilities
            comparison_data = validator.compare_volatilities(iv_results, historical_prices)
            
            # Visualize comparison
            try:
                viz_filename = validator.visualize_volatility_comparison(comparison_data)
                print(f"Volatility comparison plot saved to {viz_filename}")
            except Exception as viz_error:
                print(f"Error creating volatility visualization for {ticker}:")
                print(traceback.format_exc())
        
        except Exception as ticker_error:
            print(f"Unexpected error processing {ticker}:")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()