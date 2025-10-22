import pandas as pd
import numpy as np
import requests
import joblib
import time
import re
import os
import praw
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv2, boxscoreadvancedv2
from dotenv import load_dotenv

# 1. load env & config

# Load variables from .env file
load_dotenv()

# config (securely loaded from .env) 
YOUR_ODDS_API_KEY = os.environ.get('YOUR_ODDS_API_KEY')
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT')

# constants
# !!! update each season !!!
CURRENT_SEASON = '2025-26'
CURRENT_SEASON_YEAR = '2026' # the end year of the season

PLAYER_VALUE_URL = f"https://www.basketball-reference.com/leagues/NBA_{CURRENT_SEASON_YEAR}_advanced.html"
INJURY_URL = "https://www.rotowire.com/basketball/injury-report.php"
ODDS_API_URL = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ROLLING_WINDOW = 10

FINAL_FEATURES = [
    'DIFF_E_OFF_RATING', 'DIFF_E_DEF_RATING', 'DIFF_E_NET_RATING', 'DIFF_E_PACE',
    'DIFF_E_TM_TOV_PCT', 'DIFF_E_OREB_PCT', 'DIFF_E_AST_RATIO', 'DIFF_E_REB_PCT',
    'DIFF_INJURY_IMPACT', 'DIFF_DAYS_REST', 'DIFF_IS_BACK_TO_BACK'
]

# 2. helper functions for data fetching

def get_team_map():
    """Gets a map of NBA team names to IDs and abbreviations."""
    team_dict = teams.get_teams()
    team_map = {}
    for team in team_dict:
        # map full name, simple name, and abbreviation to ID
        team_map[team['full_name']] = team['id']
        team_map[team['nickname']] = team['id']
        team_map[team['abbreviation']] = team['id']
    
    # add common variations found in odds APIs
    team_map['LA Clippers'] = 1610612746
    team_map['Los Angeles Clippers'] = 1610612746
    return team_map

def get_current_player_value():
    """Scrapes B-Ref for *this season's* BPM to value players."""
    print("Fetching current season player values (BPM)...")
    try:
        response = requests.get(PLAYER_VALUE_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'advanced_stats'})
        df = pd.read_html(str(table))[0]
        df = df[df['Rk'] != 'Rk'].dropna(subset=['Player'])
        df['Player'] = df['Player'].apply(lambda x: re.sub(r'[*]', '', str(x)))
        df['BPM'] = pd.to_numeric(df['BPM'], errors='coerce')
        df = df.dropna(subset=['Player', 'BPM'])
        # create a simple lookup dictionary
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
        tables = pd.read_html(response.text)
        df = pd.concat([tables[0], tables[1]]) # two tables on the page
        df = df[['Player', 'Status', 'Team']]
        out_players = df[df['Status'].isin(['Out', 'Out Indefinitely'])]
        print(f"Found {len(out_players)} players confirmed OUT.")
        return out_players
    except Exception as e:
        print(f"Error scraping injuries: {e}. Using empty list.")
        return pd.DataFrame(columns=['Player', 'Status', 'Team'])

def get_todays_odds():
    """Fetches today's moneyline odds from The Odds API."""
    print("Fetching today's odds...")
    if not YOUR_ODDS_API_KEY or YOUR_ODDS_API_KEY == 'YOUR_API_KEY_HERE':
        print("ERROR: ODDS_API_KEY not found. Please add it to your .env file.")
        return []
        
    params = {'apiKey': YOUR_ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h'}
    try:
        response = requests.get(ODDS_API_URL, params=params)
        response.raise_for_status() # raise an error for bad responses
        print(f"Odds API Requests Remaining: {response.headers.get('x-requests-remaining')}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds: {e}")
        return []

def get_team_rolling_stats(team_id, player_value_map):
    """
    Fetches and calculates all rolling features for a single team.
    This is the core of the feature engineering.
    """
    print(f"Calculating rolling stats for team {team_id}...")
    try:
        # 1. get last 10 game logs
        glog = teamgamelog.TeamGameLog(
            team_id=team_id, 
            season=CURRENT_SEASON.split('-')[0], # use start year ('2025)
            season_type_all_star='Regular Season'
        )
        games_df = glog.get_data_frames()[0].head(ROLLING_WINDOW)
        
        if len(games_df) < 5: # need at least 5 games to get a decent average
            print(f"Warning: Not enough recent games for team {team_id}. Skipping.")
            return None

        game_ids = games_df['Game_ID'].tolist()
        game_stats_list = []

        # 2. loop through recent games to get advanced stats and injury impact
        for game_id in game_ids:
            adv_box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            team_stats = adv_box.get_data_frames()[1]
            team_adv_stats = team_stats[team_stats['TEAM_ID'] == team_id].iloc[0]
            
            # calculate injury impact for game
            trad_box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = trad_box.get_data_frames()[0]
            inactive_players = player_stats[
                (player_stats['MIN'].isnull()) & (player_stats['TEAM_ID'] == team_id)
            ]
            
            injury_impact = 0
            for _, player in inactive_players.iterrows():
                # look up player's BPM value
                injury_impact += player_value_map.get(player['PLAYER_NAME'], 0)
            
            # combine all stats for this one game
            game_stats = team_adv_stats.to_dict()
            game_stats['INJURY_IMPACT'] = injury_impact
            game_stats_list.append(game_stats)
            
            time.sleep(0.7) 

        # 3. create DataFrame and calculate rolling averages
        recent_stats_df = pd.DataFrame(game_stats_list)
        
        final_stats = {}
        features_to_average = [
            'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE', 
            'E_TM_TOV_PCT', 'E_OREB_PCT', 'E_AST_RATIO', 'E_REB_PCT', 'INJURY_IMPACT'
        ]
        
        for factor in features_to_average:
            final_stats[f'ROLL_{factor}'] = recent_stats_df[factor].mean()

        # 4. get situational features
        today = pd.to_datetime('today').tz_localize('America/Chicago')
        last_game_date = pd.to_datetime(games_df.iloc[0]['GAME_DATE'])
        final_stats['DAYS_REST'] = (today - last_game_date).days - 1
        final_stats['IS_BACK_TO_BACK'] = (final_stats['DAYS_REST'] == 0).astype(int)
        
        print(f"Stats calculated for team {team_id}.")
        return final_stats

    except Exception as e:
        print(f"Critical Error in get_team_rolling_stats for {team_id}: {e}")
        return None

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
        
        # search for the official game thread first
        query = f'Game Thread: {away_team} at {home_team}'
        thread = next(subreddit.search(query, sort='new', time_filter='day', limit=1), None)

        # if no game thread, fall back to daily discussion
        if not thread:
            query = 'Daily Discussion Thread'
            thread = next(subreddit.search(query, sort='new', time_filter='day', limit=1), None)

        if not thread:
            print("Could not find relevant Reddit thread.")
            return 0
            
        thread.comments.replace_more(limit=0)
        home_sentiment, away_sentiment = [], []
        
        # analyze top 300 comments
        for comment in thread.comments.list()[:300]:
            body = comment.body.lower()
            # use team nicknames for better matching
            home_nick = home_team.split(' ')[-1].lower()
            away_nick = away_team.split(' ')[-1].lower()
            
            score = analyzer.polarity_scores(body)['compound']
            
            if home_nick in body and away_nick in body:
                continue 
            if home_nick in body:
                home_sentiment.append(score)
            if away_nick in body:
                away_sentiment.append(score)
        
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
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# 3. main prediction logic

def main():
    print("--- 🏀 NBA Prediction Model v1.0 ---")
    
    # 1. load all static data 
    try:
        model = joblib.load('nba_model.pkl')
        print("Model 'nba_model.pkl' loaded successfully.")
    except FileNotFoundError:
        print("ERROR: 'nba_model.pkl' not found.")
        print("Please run '03_train_model.py' first.")
        return
    
    team_map = get_team_map()
    player_value_map = get_current_player_value()
    out_players_df = get_realtime_injuries()
    todays_odds = get_todays_odds()
    
    if not todays_odds:
        print("No games or odds found. Exiting.")
        return

    print(f"\n--- Found {len(todays_odds)} Games for Today ---")
    
    # 2. loop through each game 
    for game in todays_odds:
        home_team_name = game['home_team']
        away_team_name = game['away_team']
        game_time = pd.to_datetime(game['commence_time']).tz_convert('America/Chicago')
        
        print("\n" + "="*50)
        print(f"GAME: {away_team_name} @ {home_team_name} ({game_time.strftime('%I:%M %p %Z')})")
        
        # 3. get team IDs 
        home_team_id = team_map.get(home_team_name)
        away_team_id = team_map.get(away_team_name)
        
        if not home_team_id or not away_team_id:
            print(f"Skipping game: Could not map team names ('{home_team_name}' or '{away_team_name}')")
            continue

        # 4. get live odds 
        try:
            # find a consensus or a specific bookmaker 
            bookmaker = next(b for b in game['bookmakers'] if b['key'] == 'draftkings')
            market = next(m for m in bookmaker['markets'] if m['key'] == 'h2h')
            home_odds = next(o for o in market['outcomes'] if o['name'] == home_team_name)['price']
            away_odds = next(o for o in market['outcomes'] if o['name'] == away_team_name)['price']
        except StopIteration:
            print("Skipping game: Could not find DraftKings moneyline odds.")
            continue
            
        home_market_prob = odds_to_prob(home_odds)
        
        # 5. build feature vector 
        home_stats = get_team_rolling_stats(home_team_id, player_value_map)
        away_stats = get_team_rolling_stats(away_team_id, player_value_map)
        
        if not home_stats or not away_stats:
            print("Skipping game: Could not calculate rolling stats for both teams.")
            continue
            
        # create the differential feature dictionary
        feature_dict = {}
        for factor in FINAL_FEATURES:
            if 'INJURY_IMPACT' in factor:
                feature_dict[factor] = home_stats['ROLL_INJURY_IMPACT'] - away_stats['ROLL_INJURY_IMPACT']
            elif 'DAYS_REST' in factor:
                feature_dict[factor] = home_stats['DAYS_REST'] - away_stats['DAYS_REST']
            elif 'IS_BACK_TO_BACK' in factor:
                feature_dict[factor] = home_stats['IS_BACK_TO_BACK'] - away_stats['IS_BACK_TO_BACK']
            elif 'E_' in factor:
                # e.g., 'DIFF_E_OFF_RATING' = 'ROLL_E_OFF_RATING'
                home_key = factor.replace('DIFF_', 'ROLL_')
                away_key = factor.replace('DIFF_', 'ROLL_')
                feature_dict[factor] = home_stats[home_key] - away_stats[away_key]
        
        # 6. make prediction 
        features_df = pd.DataFrame([feature_dict], columns=FINAL_FEATURES)
        model_prob = model.predict_proba(features_df)[0][1] # [:, 1] is prob of "Home Win"
        
        # 7. get sentiment & injury filter 
        sentiment_score = get_reddit_sentiment(home_team_name, away_team_name)
        
        # check for new major injuries not in the rolling average
        home_abbr = next((t['abbreviation'] for t in teams.get_teams() if t['id'] == home_team_id), None)
        away_abbr = next((t['abbreviation'] for t in teams.get_teams() if t['id'] == away_team_id), None)
        
        new_home_injuries = out_players_df[out_players_df['Team'] == home_abbr]['Player'].tolist()
        new_away_injuries = out_players_df[out_players_df['Team'] == away_abbr]['Player'].tolist()
        
        # 8. calculate edge & make decision 
        edge = model_prob - home_market_prob
        
        print("\n--- ANALYSIS ---")
        print(f"  Market Odds: {home_team_name} ({home_odds}) -> {home_market_prob*100:.1f}%")
        print(f"  My Model:    {home_team_name} -> {model_prob*100:.1f}%")
        print(f"  Edge: {edge*100:+.2f}%")
        print(f"  Sentiment (Home-Away): {sentiment_score:+.3f}")
        if new_home_injuries: print(f"  New Home Injuries: {', '.join(new_home_injuries)}")
        if new_away_injuries: print(f"  New Away Injuries: {', '.join(new_away_injuries)}")

        # decision logic 
        bet_threshold = 0.04 # bet if edge is > 4%
        sentiment_threshold = -0.05 # don't bet if sentiment is very negative
        
        print("\n--- FINAL DECISION ---")
        if edge > bet_threshold and sentiment_score > sentiment_threshold:
            print(f"  ✅ BET: {home_team_name} (ML @ {home_odds})")
        elif edge < -bet_threshold and sentiment_score < (sentiment_threshold * -1):
            print(f"  ✅ BET: {away_team_name} (ML @ {away_odds})")
        else:
            if edge < 0.01 and edge > -0.01:
                print("  ❌ NO BET: No significant edge.")
            elif edge > bet_threshold and sentiment_score <= sentiment_threshold:
                print(f"  ❌ NO BET: Positive edge, but sentiment is too negative ({sentiment_score:.3f}).")
            elif edge < -bet_threshold and sentiment_score >= (sentiment_threshold * -1):
                print(f"  ❌ NO BET: Negative edge, but sentiment is too positive ({sentiment_score:.3f}).")
            else:
                print("  ❌ NO BET: Edge is not strong enough.")

if __name__ == "__main__":
    main()