#Sector-Rotation Economic Stage Model
#This script uses historical price data of major US sector ETFs from Yahoo Finance (via yfinance) 
#to analyze sector momentum and infer the current stage of the economic cycle. 
#It produces simple visualizations and a summary statement in plain English.

#Author: utkarshkherwal
#Date: 2025-07-19

'''ETFs Used:
- XLK (Technology)
- XLY (Consumer Discretionary)
- XLF (Financials)
- XLV (Healthcare)
- XLU (Utilities)
- XLE (Energy)
- XLRE (Real Estate)'''


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Sector ETFs ko define kara aur unki mapping economic phases ke sath krdi.

SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLY': 'Consumer Discretionary',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLU': 'Utilities',
    'XLE': 'Energy',
    'XLRE': 'Real Estate',
}


# Mapping of economic phase to strong sectors
ECONOMIC_PHASES = {
    'Recovery': ['Technology', 'Consumer Discretionary'],
    'Expansion': ['Financials', 'Industrials'],  # Industrials not present, will adjust logic
    'Slowdown': ['Healthcare', 'Consumer Staples'],  # Consumer Staples not present, will adjust
    'Recession': ['Utilities', 'Consumer Staples'],
}


# Fallback: If Consumer Staples or Industrials not in ETFs, substitute with similar sectors
SUBSTITUTES = {
    'Industrials': 'Financials',  # Use Financials as proxy for Expansion
    'Consumer Staples': 'Healthcare'  # Use Healthcare as proxy for Slowdown/Recession
}

# Download historical price data for all ETFs
# 'Close' prices are already adjusted by default in yfinance (since 2023)
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')


# List of tickers to download
# Extracting keys from SECTOR_ETFS dictionary
tickers = list(SECTOR_ETFS.keys())


# Downloading the data
etf_data = yf.download(tickers, start=start_date, end=end_date)

print(tickers)


print("\nColumns in the downloaded DataFrame:")
print(etf_data.columns)

#Extract the 'Close' prices for each ETF from the MultiIndex columns
if isinstance(etf_data.columns, pd.MultiIndex):
    if 'Close' in etf_data.columns.levels[0]:
        close_prices = etf_data['Close']
    else:
        print("\nError: 'Close' not found in MultiIndex columns.")
        print("Available first level columns:", etf_data.columns.levels[0])
        close_prices = None
elif 'Close' in etf_data.columns:
    close_prices = etf_data[['Close']]
else:
    print("\nError: 'Close' not found in columns.")
    print("Available columns:", etf_data.columns)
    close_prices = None

if close_prices is not None:
    print("\nFirst few rows of daily Close prices:")
    print(close_prices.head())
else:
    print("\nCould not extract 'Close' prices. Please check the output above for debugging.")


# Resample daily close prices to monthly frequency, using the last close of each month
monthly_close = close_prices.resample('M').last()

print("\nFirst few rows of monthly Close prices:")
print(monthly_close.head())

# Calculate 1-month % returns   and 3-month % returns
# Using pct_change to calculate percentage change over specified periods
one_month_returns = monthly_close.pct_change(periods=1) * 100  #multiplying by 100 for percent
three_month_returns = monthly_close.pct_change(periods=3) * 100  

# Get the most recent month for analysis
latest_month = monthly_close.index[-1]

latest_1m = one_month_returns.loc[latest_month]
latest_3m = three_month_returns.loc[latest_month]

# Combine 1m and 3m returns into a DataFrame for easy viewing
sector_perf = pd.DataFrame({
    '1M Return (%)': latest_1m,
    '3M Return (%)': latest_3m,
})
sector_perf.index.name = 'Sector'

sector_perf_sorted = sector_perf.sort_values(by='3M Return (%)', ascending=False)

print("\nSector performance sorted by 3-month return (latest month):")
print(sector_perf_sorted)

latest_1m = one_month_returns.loc[latest_month]
latest_3m = three_month_returns.loc[latest_month]

# Combine 1m and 3m returns into a DataFrame for easy viewing
sector_perf = pd.DataFrame({
    'Sector': [SECTOR_ETFS[ticker] for ticker in latest_1m.index],
    '1M Return (%)': latest_1m.values,
    '3M Return (%)': latest_3m.values,
})
sector_perf.set_index('Sector', inplace=True)

# Rank sectors by 3M Return (primary momentum metric)
sector_perf_sorted = sector_perf.sort_values(by='3M Return (%)', ascending=False)


# Determine likely economic phase based on sector ranking
def infer_economic_phase(sector_ranking):
    """
    Infers the most likely economic phase by checking which phase's strong sectors
    are most represented in the top 3 performing sectors.
    """
    top_sectors = sector_ranking.index[:3].tolist()
    phase_scores = {}
    for phase, strong_sectors in ECONOMIC_PHASES.items():
        # Substitute for missing sectors
        strong_sectors = [
            SUBSTITUTES.get(sector, sector) if sector not in SECTOR_ETFS.values() else sector
            for sector in strong_sectors
        ]
        # Count how many of the top sectors match
        matches = len(set(top_sectors) & set(strong_sectors))
        phase_scores[phase] = matches

    # Pick the phase with most matches in top sectors
    likely_phase = max(phase_scores, key=phase_scores.get)
    return likely_phase, phase_scores

likely_phase, phase_scores = infer_economic_phase(sector_perf_sorted)

#--------------------------------Visualization--------------------------------
# Set up the plotting style
sns.set(style="whitegrid")

# Bar chart of sector 3M returns for latest month
plt.figure(figsize=(10, 6))
sns.barplot(
    x=sector_perf_sorted.index,
    y=sector_perf_sorted['3M Return (%)'],
    palette="viridis"
)
plt.title(f"Sector 3-Month Returns ({latest_month.strftime('%b %Y')})")
plt.ylabel("3M Return (%)")
plt.xlabel("Sector")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary chart: sector ranking by 3M returns
plt.figure(figsize=(10, 6))
sns.barplot(
    x=sector_perf_sorted['3M Return (%)'],
    y=sector_perf_sorted.index,
    palette="mako"
)
plt.title("Sector Performance Rankings (3M Returns)")
plt.xlabel("3M Return (%)")
plt.ylabel("Sector")
plt.tight_layout()
plt.show()

# Show the most likely economic phase as a text label
plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(
    0.5, 0.5,
    f"Most Likely Economic Phase: {likely_phase}",
    fontsize=18,
    color="navy",
    ha='center', va='center'
)
plt.show()



# Get top 3 performing sectors
top_3_sectors = sector_perf_sorted.index[:3].tolist()

# Simple explanation for each economic phase in layman's terms
phase_explanations = {
    'Recovery': "Markets are bouncing back. Technology and consumer-focused sectors tend to lead, reflecting renewed optimism.",
    'Expansion': "The economy is growing steadily. Financials (and typically Industrials) perform well as growth broadens.",
    'Slowdown': "Growth is cooling off. Healthcare and defensive sectors gain strength as investors seek stability.",
    'Recession': "Economic activity is contracting. Utilities and defensive sectors outperform as investors prioritize safety."
}

print("\n" + "="*60)
print("Sector Rotation Model Summary")
print("="*60)
print(f"Top 3 Sectors (recent performance): {', '.join(top_3_sectors)}")
print(f"Most Likely Economic Phase: {likely_phase}")
print(f"\nWhat this means: {phase_explanations.get(likely_phase, 'No explanation available.')}")
print("="*60)

# ------------------------------
# What this script does (in super simple words):

# - It gets monthly price data for different parts of the stock market from Yahoo Finance.
# - Then, it checks how much each sector has gone up or down in the last 1 and 3 months.
# - It ranks the sectors from best to worst based on how they performed.
# - It compares the top sectors to a normal economic cycle to guess what stage the economy is in (like growth or slowdown).
# - It shows some easy-to-read charts so you can see which sectors are doing well.
# - At the end, it gives you a simple summary saying what part of the economic cycle we're likely in and what that means.

# You don’t need to know coding or economics — just look at the final summary!
# ------------------------------