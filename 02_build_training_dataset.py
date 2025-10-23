import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoreadvancedv2

# config
SEASONS = ['2022-23', '2023-24', '2024-25']
# advanced features: four Factors + AST ratio and REB Pct
ADVANCED_FEATURES = [
    'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE', 
    'E_TM_TOV_PCT', 'E_OREB_PCT', 'E_AST_RATIO', 'E_REB_PCT'
]
ROLLING_WINDOW = 10

def load_player_value():
    """Loads the BPM map we created."""
    try:
        df = pd.read_csv('player_value_map.csv', encoding='utf-8')
        # create a multi-index for fast lookups
        df['Season'] = df['Season'].apply(lambda x: f"{x-1}-{str(x)[-2:]}")
        df = df.set_index(['Season', 'Player'])
        return df['BPM'].to_dict()
    except FileNotFoundError:
        print("ERROR: 'player_value_map.csv' not found.")
        print("Please run '01_get_player_value.py' first.")
        return None

def get_game_ids():
    """Fetches all regular season game IDs."""
    print("Fetching all game IDs...")
    all_games = []
    for season in SEASONS:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',
            season_type_nullable='Regular Season'
        )
        games_df = finder.get_data_frames()[0]
        all_games.append(games_df)
        print(f"Fetched {len(games_df)} game logs for {season}.")
        time.sleep(0.6)
    
    full_game_log = pd.concat(all_games, ignore_index=True)
    game_ids = full_game_log['GAME_ID'].unique()
    print(f"Total unique games found: {len(game_ids)}")
    return game_ids, full_game_log

def get_game_data_and_injuries(game_ids, player_value_map):
    """
    Fetches advanced stats and calculates injury impact for every game.
    """
    print(f"Fetching data for {len(game_ids)} games. This will take a while...")
    all_team_data = []
    
    for i, game_id in enumerate(game_ids):
        if i % 100 == 0:
            print(f"Processing game {i} of {len(game_ids)}...")
        
        try:
            # 1. get advanced team stats
            adv_box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            team_stats_adv = adv_box.get_data_frames()[1]
            season = adv_box.get_data_frames()[0]['SEASON'] # get season 
            
            # 2. get traditional box score (for inactive players)
            trad_box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = trad_box.get_data_frames()[0]
            
            # 3. calculate injury impact
            inactive_players = player_stats[player_stats['MIN'].isnull()]
            injury_impact = {team_stats_adv.iloc[0]['TEAM_ID']: 0, team_stats_adv.iloc[1]['TEAM_ID']: 0}
            
            for _, player in inactive_players.iterrows():
                player_name = player['PLAYER_NAME']
                team_id = player['TEAM_ID']
                
                # look up player's BPM
                player_bpm = player_value_map.get((season, player_name), 0) # default to 0 if not found
                injury_impact[team_id] += player_bpm
                
            team_stats_adv['INJURY_IMPACT'] = team_stats_adv['TEAM_ID'].map(injury_impact)
            all_team_data.append(team_stats_adv)
            
            time.sleep(0.7) 
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            
    return pd.concat(all_team_data, ignore_index=True)

def create_final_dataset(game_stats_df, all_games_log):
    """Engineers rolling features and merges into a final model input."""
    print("Engineering rolling features...")
    df = game_stats_df.sort_values(by='GAME_ID').copy()
    
    # create rolling averages
    features_to_roll = ADVANCED_FEATURES + ['INJURY_IMPACT']
    for factor in features_to_roll:
        df[f'ROLL_{factor}'] = df.groupby('TEAM_ID')[factor].transform(
            lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=5).mean()
        )
    
    # add situational features (rest, B2B)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_ID'].str[:10], format='%Y-%m-%d')
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days - 1
    df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 0).astype(int)
    
    # drop initial games without full rolling data
    df = df.dropna(subset=[f'ROLL_{ADVANCED_FEATURES[0]}'])
    
    print("Merging Home/Away data...")
    # get WL info from the original game finder
    wl_df = all_games_log[['GAME_ID', 'TEAM_ID', 'MATCHUP', 'WL']]
    df = df.merge(wl_df, on=['GAME_ID', 'TEAM_ID'])
    
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
    home_df = df[df['IS_HOME'] == 1].add_prefix('HOME_')
    away_df = df[df['IS_HOME'] == 0].add_prefix('AWAY_')
    
    final_df = pd.merge(home_df, away_df, left_on='HOME_GAME_ID', right_on='AWAY_GAME_ID')
    
    # create final differential features
    for factor in ADVANCED_FEATURES:
        final_df[f'DIFF_{factor}'] = final_df[f'HOME_ROLL_{factor}'] - final_df[f'AWAY_ROLL_{factor}']
    
    final_df['DIFF_INJURY_IMPACT'] = final_df['HOME_ROLL_INJURY_IMPACT'] - final_df['AWAY_ROLL_INJURY_IMPACT']
    final_df['DIFF_DAYS_REST'] = final_df['HOME_DAYS_REST'] - final_df['AWAY_DAYS_REST']
    final_df['DIFF_IS_BACK_TO_BACK'] = final_df['HOME_IS_BACK_TO_BACK'] - final_df['AWAY_IS_BACK_TO_BACK']
    
    # define the target variable
    final_df['HOME_TEAM_WON'] = (final_df['HOME_WL'] == 'W').astype(int)
    
    # save final dataset
    final_df.to_csv('nba_model_dataset.csv', index=False)
    print(f"Dataset complete. Saved to nba_model_dataset.csv. Shape: {final_df.shape}")

def main():
    player_value_map = load_player_value()
    if player_value_map:
        game_ids, all_games_log = get_game_ids()
        game_stats_df = get_game_data_and_injuries(game_ids, player_value_map)
        create_final_dataset(game_stats_df, all_games_log)

if __name__ == "__main__":
    main()