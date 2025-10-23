import requests
import pandas as pd
import time
import re

SEASONS = [2022, 2023, 2024, 2025] # going back a few years to present
BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html"

def scrape_player_value():
    """
    Scrapes Basketball-Reference for advanced player stats (BPM)
    for multiple seasons using pandas.read_html with explicit UTF-8 decoding.
    """
    print("Starting player value scraping using pandas.read_html...")
    all_player_data = []

    for season in SEASONS:
        url = BASE_URL.format(season)
        print(f"Fetching data for {season} season...")

        try:
            # use headers to mimic a browser
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status() # Raise an exception for bad status codes

            # explicitly decode the content using UTF-8
            html_content = response.content.decode('utf-8')

            # pass the explicitly decoded HTML string to pandas
            tables = pd.read_html(html_content)

            # find the correct table by looking for 'BPM' column
            found_table = None
            for df_table in tables:
                # check if 'BPM' exists in the columns (might be multi-level)
                if any('BPM' in col for col in df_table.columns):
                    found_table = df_table
                    break 

            if found_table is not None:
                df = found_table.copy() # Work on a copy

                # handle potential multi-level headers more robustly
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

                # find columns case-insensitively
                player_col = next((col for col in df.columns if 'Player' in col), None)
                bpm_col = next((col for col in df.columns if 'BPM' in col), None)

                if not player_col or not bpm_col:
                    print(f"Could not find 'Player' or 'BPM' columns in the found table for {season}.")
                    continue # Skip this season

                # select and rename
                df = df[[player_col, bpm_col]].copy()
                df.columns = ['Player', 'BPM']

                # clean header rows and player names
                df = df[df['Player'] != 'Player'].copy() 
                df['Player'] = df['Player'].apply(lambda x: re.sub(r'[*]', '', str(x)))
                df['BPM'] = pd.to_numeric(df['BPM'], errors='coerce')
                df = df.dropna(subset=['Player', 'BPM'])

                df['Season'] = season
                all_player_data.append(df)
                print(f"Successfully fetched and parsed data for {season}.")
            else:
                # pd.read_html didn't find a table with a 'BPM' column
                print(f"Could not find a table with 'BPM' column for {season}.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        except ValueError as e:
            # if pd.read_html finds NO tables at all
            print(f"Error parsing tables for {season}: {e}. Maybe no tables found?")
        except Exception as e: # catch other potential errors
            print(f"An unexpected error occurred for {season}: {e}")

        time.sleep(4)

    # combine all seasons and save if data was collected
    if all_player_data:
        final_df = pd.concat(all_player_data, ignore_index=True)
        # ensure BPM is numeric before final dropna
        final_df['BPM'] = pd.to_numeric(final_df['BPM'], errors='coerce')
        final_df = final_df.dropna(subset=['Player', 'BPM'])

        # save to CSV using UTF-8 encoding
        final_df.to_csv('player_value_map.csv', index=False, encoding='utf-8')
        print("\nSuccessfully saved all player values to 'player_value_map.csv' using UTF-8 encoding.")
        print("\n--- Final DataFrame Head ---")
        print(final_df.head())

    else:
        print("\nNo data was scraped. Exiting.")


if __name__ == "__main__":
    scrape_player_value()