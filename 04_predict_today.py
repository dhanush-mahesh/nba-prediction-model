import pandas as pd
import numpy as np
import requests
import joblib
import time
import re
import os
import praw
from bs4 import BeautifulSoup, Comment # <-- Import Comment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nba_api.stats.static import teams
# We no longer need teamgamelog
from dotenv import load_dotenv
from io import StringIO 

# --- 1. LOAD ENVIRONMENT & CONFIG ---
load_dotenv()
YOUR_ODDS_API_KEY = os.environ.get('YOUR_ODDS_API_KEY')
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT')

# --- CONSTANTS ---
CURRENT_SEASON = '2025-26'
CURRENT_SEASON_YEAR = '2026'
PLAYER_VALUE_URL = f"https://www.basketball-reference.com/leagues/NBA_{CURRENT_SEASON_YEAR}_advanced.html"
INJURY_URL = "https://www.rotowire.com/basketball/injury-report.php"
ODDS_API_URL = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ROLLING_WINDOW = 10

FINAL_FEATURES = [
    'DIFF_FG%', 'DIFF_3P%', 'DIFF_FT%', 'DIFF_OREB',
    'DIFF_DREB', 'DIFF_AST', 'DIFF_TOV', 'DIFF_STL', 'DIFF_BLK',
    'DIFF_DAYS_REST', 'DIFF_IS_BACK_TO_BACK'
]
FEATURES_TO_ROLL = [
    'FG%', '3P%', 'FT%', 'OREB', 
    'DREB', 'AST', 'TOV', 'STL', 'BLK'
]
BASKETBALL_REF_COL_MAP = {
    'FG%': 'FG%',
    '3P%': '3P%',
    'FT%': 'FT%',
    'ORB': 'OREB', # B-Ref uses 'ORB'
    'DRB': 'DREB', # B-Ref uses 'DRB'
    'AST': 'AST',
    'TOV': 'TOV',
    'STL': 'STL',
    'BLK': 'BLK'
}


# --- 2. HELPER FUNCTIONS (DATA FETCHING) ---

def get_team_map():
    """Gets a map of NBA team names to IDs."""
    team_dict = teams.get_teams()
    team_map = {}
    for team in team_dict:
        team_map[team['full_name']] = {'id': team['id'], 'abbr': team['abbreviation']}
        team_map[team['nickname']] = {'id': team['id'], 'abbr': team['abbreviation']}
    team_map['LA Clippers'] = {'id': 1610612746, 'abbr': 'LAC'}
    return team_map

def get_current_player_value():
    """Scrapes B-Ref for *this season's* BPM to value players."""
    print("Fetching current season player values (BPM)...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(PLAYER_VALUE_URL, headers=headers)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
        
        # --- FIX: Find table inside comments ---
        soup = BeautifulSoup(html_content, 'html.parser')
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        table_html = ""
        for comment in comments:
            if 'id="advanced_stats"' in comment:
                table_html = comment
                break

        if not table_html:
            print("Could not find BPM table comment on B-Ref page.")
            return {}

        tables = pd.read_html(StringIO(table_html))
        
        if not tables:
            print("No tables found in BPM comment.")
            return {}

        df = tables[0].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
        
        player_col = next((col for col in df.columns if 'Player' in col), None)
        bpm_col = next((col for col in df.columns if 'BPM'in col), None)
        
        if not player_col or not bpm_col:
             print("Could not find Player/BPM columns in table.")
             return {}

        df = df[[player_col, bpm_col]].copy()
        df.columns = ['Player', 'BPM']
        df = df[df['Player'] != 'Player'].copy()
        df['Player'] = df['Player'].apply(lambda x: re.sub(r'[*]', '', str(x)))
        df['BPM'] = pd.to_numeric(df['BPM'], errors='coerce')
        df = df.dropna(subset=['Player', 'BPM'])
        
        player_value_map = df.set_index('Player')['BPM'].to_dict()
        print(f"Successfully fetched BPM for {len(player_value_map)} players.")
        return player_value_map
    except Exception as e:
        print(f"Error fetching current player value: {e}. Using empty map.")
        return {}

def get_realtime_injuries():
    """Scrapes RotoWire for today's injury list for filtering."""
    print("Fetching real-time injuries from RotoWire...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(INJURY_URL, headers=headers)
        response.raise_for_status()
        
        # --- FIX: Use `match` to find tables containing 'Status' ---
        # This is more robust than class names
        tables = pd.read_html(StringIO(response.text), match='Status')
        
        if not tables:
             print("No tables with 'Status' found on RotoWire. Injury list will be empty.")
             return {}

        df = pd.concat(tables) # Concat all found tables (usually 2)
        df = df[['Player', 'Status', 'Team']]
        out_players = df[df['Status'].isin(['Out', 'Out Indefinitely'])]
        injury_dict = out_players.groupby('Team')['Player'].apply(list).to_dict()
        print(f"Found {len(out_players)} players confirmed OUT across the league.")
        return injury_dict
    except Exception as e:
        print(f"Error scraping injuries: {e}. Using empty list.")
        return {}

def get_todays_odds():
    """Fetches today's moneyline odds from The Odds API."""
    print("Fetching today's odds...")
    if not YOUR_ODDS_API_KEY or YOUR_ODDS_API_KEY == 'YOUR_API_KEY_HERE':
        print("ERROR: ODDS_API_KEY not found. Please add it to your .env file.")
        return []
        
    params = {'apiKey': YOUR_ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h'}
    try:
        response = requests.get(ODDS_API_URL, params=params)
        response.raise_for_status() 
        print(f"Odds API Requests Remaining: {response.headers.get('x-requests-remaining')}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds: {e}")
        return []

# --- THIS ENTIRE FUNCTION IS UPDATED ---
def get_team_rolling_stats(team_abbr):
    """
    Fetches a team's recent stats from BASKETBALL-REFERENCE
    by reading the HTML comments.
    """
    print(f"Calculating rolling stats for team {team_abbr} from Basketball-Reference...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/{CURRENT_SEASON_YEAR}/gamelog/"
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Will catch 500 Server Errors here
        html_content = response.content.decode('utf-8')
        
        # --- FIX: Find the gamelog table inside the HTML comments ---
        soup = BeautifulSoup(html_content, 'html.parser')
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        table_html = ""
        for comment in comments:
            if 'id="tgl_basic"' in comment:
                table_html = comment
                break
        
        if not table_html:
            print(f"Could not find gamelog table comment for {team_abbr}.")
            return None

        tables = pd.read_html(StringIO(table_html), attrs={'id': 'tgl_basic'})
        if not tables:
            print(f"Could not parse gamelog table for {team_abbr}.")
            return None
        
        df = tables[0]
        
        # Clean the table: Remove header rows (where 'G' == 'G')
        df = df[df['G'] != 'G'].copy()
        
        # Get the most recent games (top of the table)
        df = df.head(ROLLING_WINDOW)

        if len(df) < 5: 
            print(f"Warning: Not enough recent games for team {team_abbr}. Skipping.")
            return None

        # 3. Create DataFrame and calculate rolling averages
        final_stats = {}
        
        for feature_name, b_ref_col in BASKETBALL_REF_COL_MAP.items():
            df[b_ref_col] = pd.to_numeric(df[b_ref_col], errors='coerce')
            final_stats[f'ROLL_{feature_name}'] = df[b_ref_col].mean()

        # 4. Get Situational Features
        today = pd.to_datetime('today').tz_localize('America/Chicago')
        df['Date'] = pd.to_datetime(df['Date'])
        last_game_date = df.iloc[0]['Date'] # First row is the most recent game
        
        final_stats['DAYS_REST'] = (today - last_game_date).days - 1
        final_stats['IS_BACK_TO_BACK'] = (final_stats['DAYS_REST'] == 0).astype(int)
        
        print(f"Stats calculated for team {team_abbr}.")
        return final_stats

    except Exception as e:
        print(f"Critical Error in get_team_rolling_stats for {team_abbr}: {e}")
        return None
# --- END UPDATED FUNCTION ---


def get_reddit_sentiment(home_team, away_team):
    """Gets sentiment from r/nba Daily Thread as a filter."""
    print(f"Getting Reddit sentiment for {home_team} vs {away_team}...")
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        print("Warning: Reddit credentials not found in .env. Skipping sentiment.")
        return 0
        
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        analyzer = SentimentIntensityAnalyzer()
        subreddit = reddit.subreddit("nba")
        
        query = f'Game Thread: {away_team} at {home_team}'
        thread = next(subreddit.search(query, sort='new', time_filter='day', limit=1), None)

        if not thread:
            query = 'Daily Discussion Thread'
            thread = next(subreddit.search(query, sort='new', time_filter='day', limit=1), None)

        if not thread:
            print("Could not find relevant Reddit thread.")
            return 0
            
        thread.comments.replace_more(limit=0)
        home_sentiment, away_sentiment = [], []
        
        for comment in thread.comments.list()[:300]:
            body = comment.body.lower()
            home_nick = home_team.split(' ')[-1].lower()
            away_nick = away_team.split(' ')[-1].lower()
            score = analyzer.polarity_scores(body)['compound']
            
            if home_nick in body and away_nick in body: continue
            if home_nick in body: home_sentiment.append(score)
            if away_nick in body: away_sentiment.append(score)
        
        avg_home = np.mean(home_sentiment) if home_sentiment else 0
        avg_away = np.mean(away_sentiment) if away_sentiment else 0
        sentiment_diff = avg_home - avg_away
        
        print(f"Sentiment: {home_team} ({avg_home:.3f}) vs {away_team} ({avg_away:.3f}) = {sentiment_diff:+.3f}")
        return sentiment_diff
        
    except Exception as e:
        print(f"Error getting sentiment: {e}")
        return 0

def odds_to_prob(odds):
    """Converts American odds to implied probability."""
    if odds > 0: return 100 / (odds + 100)
    else: return abs(odds) / (abs(odds) + 100)

# --- 3. MAIN PREDICTION LOGIC ---

def main():
    print("--- 🏀 NBA Prediction Model v2.2 (B-Ref Comment Scraper) ---")
    
    # --- 1. Load All Static Data ---
    try:
        model = joblib.load('nba_model.pkl')
        print("Model 'nba_model.pkl' loaded successfully.")
    except FileNotFoundError:
        print("ERROR: 'nba_model.pkl' not found.")
        print("Please run '03_train_model.py' first.")
        return
    
    team_map = get_team_map()
    player_value_map = get_current_player_value()
    injury_dict = get_realtime_injuries()
    todays_odds = get_todays_odds()
    
    if not todays_odds:
        print("No games or odds found. Exiting.")
        return

    print(f"\n--- Found {len(todays_odds)} Games for Today ---")
    
    # --- 2. Loop Through Each Game ---
    for game in todays_odds:
        home_team_name = game['home_team']
        away_team_name = game['away_team']
        game_time = pd.to_datetime(game['commence_time']).tz_convert('America/Chicago')
        
        print("\n" + "="*50)
        print(f"GAME: {away_team_name} @ {home_team_name} ({game_time.strftime('%I:%M %p %Z')})")
        
        # --- 3. Get Team IDs & Abbrs ---
        home_team_info = team_map.get(home_team_name)
        away_team_info = team_map.get(away_team_name)
        
        if not home_team_info or not away_team_info:
            print(f"Skipping game: Could not map team names ('{home_team_name}' or '{away_team_name}')")
            continue
            
        home_team_abbr = home_team_info['abbr']
        away_team_abbr = away_team_info['abbr']

        # --- 4. Get Live Odds ---
        try:
            bookmaker = next(b for b in game['bookmakers'] if b['key'] == 'draftkings')
            market = next(m for m in bookmaker['markets'] if m['key'] == 'h2h')
            home_odds = next(o for o in market['outcomes'] if o['name'] == home_team_name)['price']
            away_odds = next(o for o in market['outcomes'] if o['name'] == away_team_name)['price']
        except StopIteration:
            print("Skipping game: Could not find DraftKings moneyline odds.")
            continue
            
        home_market_prob = odds_to_prob(home_odds)
        
        # --- 5. Build Feature Vector ---
        home_stats = get_team_rolling_stats(home_team_abbr)
        away_stats = get_team_rolling_stats(away_team_abbr)
        
        if not home_stats or not away_stats:
            print("Skipping game: Could not calculate rolling stats for both teams.")
            continue
            
        # Create the differential feature dictionary
        feature_dict = {}
        for factor in FEATURES_TO_ROLL: # Use FEATURES_TO_ROLL which maps to B-Ref cols
            feature_dict[f'DIFF_{factor}'] = home_stats[f'ROLL_{factor}'] - away_stats[f'ROLL_{factor}']
        
        feature_dict['DIFF_DAYS_REST'] = home_stats['DAYS_REST'] - away_stats['DAYS_REST']
        feature_dict['DIFF_IS_BACK_TO_BACK'] = home_stats['IS_BACK_TO_BACK'] - away_stats['IS_BACK_TO_BACK']
        
        # --- 6. Make Prediction ---
        features_df = pd.DataFrame([feature_dict], columns=FINAL_FEATURES)
        model_prob = model.predict_proba(features_df)[0][1] # [:, 1] is prob of "Home Win"
        
        # --- 7. Get Sentiment & Injury Filters ---
        sentiment_score = get_reddit_sentiment(home_team_name, away_team_name)
        
        # Get injury lists from our dictionary
        home_injuries = injury_dict.get(home_team_abbr, [])
        away_injuries = injury_dict.get(away_team_abbr, [])
        
        # Calculate BPM impact
        home_bpm_lost = sum(player_value_map.get(player, 0) for player in home_injuries)
        away_bpm_lost = sum(player_value_map.get(player, 0) for player in away_injuries)
        injury_impact_diff = home_bpm_lost - away_bpm_lost # Positive = Home team hurt more

        # --- 8. Calculate Edge & Make Decision ---
        edge = model_prob - home_market_prob
        
        print("\n--- ANALYSIS ---")
        print(f"  Market Odds: {home_team_name} ({home_odds}) -> {home_market_prob*100:.1f}%")
        print(f"  My Model:    {home_team_name} -> {model_prob*100:.1f}%")
        print(f"  Edge: {edge*100:+.2f}%")
        
        print("\n--- FILTERS ---")
        print(f"  Injury Impact (Home Lost - Away Lost): {injury_impact_diff:+.2f} BPM")
        if home_injuries: print(f"    Home Out: {', '.join(home_injuries)}")
        if away_injuries: print(f"    Away Out: {', '.join(away_injuries)}")
        print(f"  Sentiment (Home - Away): {sentiment_score:+.3f}")
        
        # Decision Logic 
        bet_threshold = 0.04 # Bet if edge is > 4%
        sentiment_threshold = -0.05 # Don't bet if sentiment is very negative
        injury_threshold = -1.5 # Don't bet home if they are missing > 1.5 BPM *more* than away team
        
        print("\n--- FINAL DECISION ---")
        if edge > bet_threshold and sentiment_score > sentiment_threshold and injury_impact_diff < (injury_threshold * -1):
            print(f"  ✅ BET: {home_team_name} (ML @ {home_odds})")
            print("     (Reason: Positive edge, good sentiment, no major injury disadvantage)")
        elif edge < -bet_threshold and sentiment_score < (sentiment_threshold * -1) and injury_impact_diff > injury_threshold:
            print(f"  ✅ BET: {away_team_name} (ML @ {away_odds})")
            print("     (Reason: Positive edge, good sentiment, no major injury disadvantage)")
        else:
            print("  ❌ NO BET: Conditions not met.")
            if edge <= bet_threshold and edge >= -bet_threshold:
                print("     (Reason: Edge is not strong enough)")
            elif sentiment_score <= sentiment_threshold:
                 print("     (Reason: Sentiment is too negative)")
            elif injury_impact_diff >= (injury_threshold * -1):
                 print(f"     (Reason: Home team injury impact ({injury_impact_diff:+.2f}) is too high)")
            elif injury_impact_diff <= injury_threshold:
                 print(f"     (Reason: Away team injury impact ({injury_impact_diff:+.2f}) is too high)")


if __name__ == "__main__":
    main()