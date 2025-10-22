import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re

SEASONS = [2022, 2023, 2024, 2025, 2026] # going back a few years to present
BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html"

def scrape_player_value():
    """
    Scrapes Basketball-Reference for advanced player stats (BPM)
    for multiple seasons.
    """
    print("Starting player value scraping...")
    all_player_data = []

    for season in SEASONS:
        url = BASE_URL.format(season)
        print(f"Fetching data for {season} season...")
        
        try:
            response = requests.get(url)
            response.raise_for_status() # exception for bad status codes
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'advanced_stats'})
            
            if table:
                # pandas to read the HTML table
                df = pd.read_html(str(table))[0]
                
                # clean the data
                df = df[df['Rk'] != 'Rk'].dropna(subset=['Player']) # remove header rows
                df = df[['Player', 'BPM']] 
                
                # clean player names 
                df['Player'] = df['Player'].apply(lambda x: re.sub(r'[*]', '', str(x)))
                
                df['Season'] = season # add season column
                all_player_data.append(df)
                print(f"Successfully fetched and parsed data for {season}.")
            else:
                print(f"Could not find advanced stats table for {season}.")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            
        time.sleep(3) 

    # combine all seasons and save
    if all_player_data:
        final_df = pd.concat(all_player_data, ignore_index=True)
        final_df['BPM'] = pd.to_numeric(final_df['BPM'], errors='coerce')
        final_df = final_df.dropna()
        
        # save to CSV
        final_df.to_csv('player_value_map.csv', index=False)
        print("\nSuccessfully saved all player values to 'player_value_map.csv'")
        print(final_df.head())
    else:
        print("No data was scraped. Exiting.")

if __name__ == "__main__":
    scrape_player_value()