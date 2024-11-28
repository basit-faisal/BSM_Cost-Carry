import pandas as pd
from datetime import datetime
import numpy as np

# Load the spot price (S&P 500 index) data with explicit date format
spot_df = pd.read_csv("S&P500_Historical.csv", parse_dates=[0], dayfirst=False, date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y'))
spot_df.columns = ["Date", "SpotPrice"]

# Remove commas from the SpotPrice column and convert to float
spot_df["SpotPrice"] = spot_df["SpotPrice"].replace({',': ''}, regex=True).astype(float)

# Load the futures price data with explicit date format
futures_df = pd.read_csv("S&P 500 Futures Historical Data.csv", parse_dates=[0], dayfirst=False, date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y'))
futures_df.columns = ["Date", "FuturesPrice"]

# Remove commas from the FuturesPrice column and convert to float
futures_df["FuturesPrice"] = futures_df["FuturesPrice"].replace({',': ''}, regex=True).astype(float)

# Expiration dates (for 2022, 2023, and 2024)
expiration_dates = {
    2022: ["03/18/2022", "06/17/2022", "09/16/2022", "12/16/2022"],
    2023: ["03/17/2023", "06/16/2023", "09/15/2023", "12/15/2023"],
    2024: ["03/15/2024", "06/21/2024", "09/20/2024", "12/20/2024"]
}

# Flatten expiration dates for easy lookup
all_expirations = [datetime.strptime(date, "%m/%d/%Y") for dates in expiration_dates.values() for date in dates]

# Assume an interest rate (example: 3% per year)
interest_rate = 0.03

# Function to calculate the theoretical futures price
def calculate_theoretical_price(spot_price, current_date, expiration_date, r=interest_rate):
    # Time to expiration in days
    T = (expiration_date - current_date).days
    # Calculate theoretical future price
    return spot_price * (1 + r * T / 365)

# Merge the spot and futures dataframes on Date
merged_df = pd.merge(spot_df, futures_df, on="Date")

# Calculate the theoretical futures prices for each row
theoretical_prices = []
for index, row in merged_df.iterrows():
    # Find the nearest expiration date after the current date
    expiration_date = min([exp for exp in all_expirations if exp >= row["Date"]])
    
    # Calculate the theoretical price
    theoretical_price = calculate_theoretical_price(row["SpotPrice"], row["Date"], expiration_date)
    theoretical_prices.append(theoretical_price)

merged_df["TheoreticalPrice"] = theoretical_prices

# Calculate the price difference
merged_df["PriceDifference"] = merged_df["FuturesPrice"] - merged_df["TheoreticalPrice"]

# Print the results for comparison
print(merged_df[["Date", "SpotPrice", "FuturesPrice", "TheoreticalPrice", "PriceDifference"]])

# Optionally, save the results to a CSV for further analysis
merged_df.to_csv("theoretical_vs_actual_futures.csv", index=False)
