import pickle
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler


def load_sp100_companies():
    """
    Load the list of S&P 100 companies from Wikipedia.

    Returns:
        pd.DataFrame: DataFrame with the list of S&P 100 companies.
    """
    print(" - Loading S&P 100 companies list")
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "wikitable sortable"})
    sp100_companies = pd.read_html(str(table))[0]
    return sp100_companies


def load_sp100_index_price(start_date, end_date):
    """
    Load historical adjusted close prices for the S&P 100 index.

    Args:
        start_date (str): Start date for fetching historical prices (YYYY-MM-DD).
        end_date (str): End date for fetching historical prices (YYYY-MM-DD).

    Returns:
        pd.Series: Series with adjusted close prices for the S&P 100 index.
    """
    print(" - Loading S&P 100 index prices")
    sp100_ticker = "^OEX"  # S&P 100 ticker on Yahoo Finance
    data = yf.download(sp100_ticker, start=start_date, end=end_date)
    return data["Adj Close"]


def load_company_data(companies, start_date, end_date):
    """
    Load historical stock data for a list of companies.

    Args:
        companies (pd.DataFrame): DataFrame with list of companies and their symbols.
        start_date (str): Start date for fetching historical data (YYYY-MM-DD).
        end_date (str): End date for fetching historical data (YYYY-MM-DD).

    Returns:
        dict: Dictionary with company symbols as keys and their historical data as values.
    """
    print(" - Loading S&P 100 companies data")
    company_data = {}
    for symbol in companies["Symbol"]:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                hist["Symbol"] = symbol
                company_data[symbol] = hist
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    return company_data


def create_company_features(company_data, sp100_index_price, sp100_companies):
    """
    Create features for each company combining stock data and sector information.

    Args:
        company_data (dict): Dictionary with company symbols as keys and historical data as values.
        sp100_index_price (pd.Series): Series with adjusted close prices for the S&P 100 index.
        sp100_companies (pd.DataFrame): DataFrame with list of S&P 100 companies and their sectors.

    Returns:
        pd.DataFrame: DataFrame with features including symbol, date, sector, and stock data.
    """
    print(" - Combining data")
    company_features = []
    
    for symbol, data in company_data.items():
        try:
            
            sector = sp100_companies.loc[sp100_companies["Symbol"] == symbol, "Sector"].values[0]
            
            data.index = data.index.strftime("%Y-%m-%d")
            
            # Combine company data with sector and SP100 price data
            for date in data.index:
                if date in sp100_index_price.index:
                    features = data.loc[date, ["Open", "High", "Low", "Close", "Volume"]].values
                    target = sp100_index_price.loc[date]
                    company_features.append([symbol, date, sector] + list(features) + [target])
                    
        except Exception as e:
            print(f"Error processing data for {symbol}: {str(e)}")
            continue

    columns = ["Symbol", "Date", "Sector", "Open", "High", "Low", "Close", "Volume", "SP100_Target"]
    features_df = pd.DataFrame(company_features, columns=columns)
    
    return features_df


def load_data(start_date, end_date):
    """
    Load and process data for S&P 100 companies and their features.

    Args:
        start_date (str): Start date for fetching historical data (YYYY-MM-DD).
        end_date (str): End date for fetching historical data (YYYY-MM-DD).

    Returns:
        tuple: Tuple containing:
            - pd.DataFrame: DataFrame with list of S&P 100 companies.
            - pd.Series: Series with adjusted close prices for the S&P 100 index.
            - pd.DataFrame: DataFrame with features for each company.
    """
    sp100_companies = load_sp100_companies()
    sp100_index_price = load_sp100_index_price(start_date, end_date)
    company_data = load_company_data(sp100_companies, start_date, end_date)
    company_features = create_company_features(company_data, sp100_index_price, sp100_companies)
    return sp100_companies, sp100_index_price, company_features


def create_sector_stats_table(company_features):
    """
    Create sector statistics table from company features.

    Args:
        company_features (pd.DataFrame): DataFrame with features for each company.

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics by sector and date.
    """
    print(" - Creating features")
    columns_to_aggregate = ["Date", "Sector", "Open", "High", "Low", "Close", "Volume"]
    
    sector_stats = company_features[columns_to_aggregate].groupby(["Date", "Sector"]).agg({
        "Open": "mean",
        "High": "mean",
        "Low": "mean",
        'Close': "mean",
        "Volume": "sum"
    }).reset_index()

    sector_stats = sector_stats.pivot(index="Date", columns="Sector").fillna(0)
    sector_stats.columns = ['{}_{}'.format(col[1], col[0]) for col in sector_stats.columns]
    
    sp100_target = company_features[["Date", "SP100_Target"]].drop_duplicates().set_index("Date")
    sector_stats["SP100_Target"] = sp100_target["SP100_Target"]
    
    return sector_stats.reset_index()


def main():
    start_date = "2019-01-01"
    end_date = "2024-01-01"
    sp100_companies, sp100_index_price, company_features = load_data(start_date, end_date)

    sector_stats= create_sector_stats_table(company_features)
    features = sector_stats.drop(["Date", "SP100_Target"], axis=1).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    data_to_save = {
        "sp100_companies": sp100_companies,
        "sp100_index_price": sp100_index_price,
        "company_features": company_features,
        "scaled_features": scaled_features
        }

    for name, data in data_to_save.items():
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f" - Successfully saved as {name}.pkl") 


if __name__ == "__main__":
    main()
