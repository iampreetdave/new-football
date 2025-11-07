"""
FOOTBALL PREDICTOR - FOTTYY CONFIDENCE METHOD
- 60% Base Confidence (from max probability)
- 40% Margin Confidence (gap between 1st and 2nd predictions)
- Validates and filters out games with invalid odds
- Generates detailed CSV output with all predictions and confidence levels
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from datetime import datetime
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
GPU_PARAMS = {'device': 'cuda', 'tree_method': 'hist'}

class DataLoader:
    def load(self, filepath):
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        
        df = pd.read_csv(filepath).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        df['date'] = pd.to_datetime(df['date_unix'], unit='s', errors='coerce')
        df = df.dropna(subset=['date'])
        
        df['home_team'] = df['home_name'].str.strip()
        df['away_team'] = df['away_name'].str.strip()
        df['league'] = df.get('fetched_league_name', 'Unknown').fillna('Unknown')
        
        # Target variables
        df['home_goals'] = pd.to_numeric(df['homeGoalCount'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['awayGoalCount'], errors='coerce')
        df = df.dropna(subset=['home_goals', 'away_goals'])
        df['total_goals'] = df['home_goals'] + df['away_goals']
        
        # PRE-MATCH xG only
        df['pre_home_xg'] = pd.to_numeric(df['team_a_xg_prematch'], errors='coerce')
        df['pre_away_xg'] = pd.to_numeric(df['team_b_xg_prematch'], errors='coerce')
        df = df.dropna(subset=['pre_home_xg', 'pre_away_xg'])
        df = df[(df['pre_home_xg'] > 0) & (df['pre_away_xg'] > 0)]
        
        df['pre_total_xg'] = df['pre_home_xg'] + df['pre_away_xg']
        df['xg_diff'] = df['pre_home_xg'] - df['pre_away_xg']
        df['xg_ratio'] = df['pre_home_xg'] / (df['pre_away_xg'] + 0.01)
        
        # PRE-MATCH PPG only
        df['home_ppg'] = pd.to_numeric(df['pre_match_home_ppg'], errors='coerce')
        df['away_ppg'] = pd.to_numeric(df['pre_match_away_ppg'], errors='coerce')
        df = df.dropna(subset=['home_ppg', 'away_ppg'])
        df['ppg_diff'] = df['home_ppg'] - df['away_ppg']
        
        # CTMCL for O/U 2.5
        if 'odds_ft_over25' in df.columns:
            df['odds_over25'] = pd.to_numeric(df['odds_ft_over25'], errors='coerce')
        else:
            df['odds_over25'] = 2.0
            
        if 'odds_ft_under25' in df.columns:
            df['odds_under25'] = pd.to_numeric(df['odds_ft_under25'], errors='coerce')
        else:
            df['odds_under25'] = 2.0
        
        # CRITICAL: Remove games with invalid O/U odds
        print(f"  Before odds filtering: {len(df)} matches")
        df = df.dropna(subset=['odds_over25', 'odds_under25'])
        df = df[(df['odds_over25'] > 1.01) & (df['odds_over25'] < 50)]
        df = df[(df['odds_under25'] > 1.01) & (df['odds_under25'] < 50)]
        print(f"  After O/U odds filtering: {len(df)} matches")
        
        # Calculate CTMCL
        df['CTMCL'] = 2.5 + (1 / df['odds_over25'] - 0.5)
        df = df[(df['CTMCL'] > 0) & (df['CTMCL'] < 10)]
        
        # Market potentials
        for col in ['o25_potential', 'o35_potential', 'o45_potential', 'btts_potential']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50)
            else:
                df[col] = 50
        
        # Moneyline Odds
        for old, new in [('odds_ft_1', 'odds_home_win'), ('odds_ft_2', 'odds_away_win')]:
            if old in df.columns:
                df[new] = pd.to_numeric(df[old], errors='coerce')
            else:
                df[new] = 2.5
        
        # CRITICAL: Remove games with invalid ML odds
        print(f"  Before ML odds filtering: {len(df)} matches")
        df = df.dropna(subset=['odds_home_win', 'odds_away_win'])
        df = df[(df['odds_home_win'] > 1.01) & (df['odds_home_win'] < 50)]
        df = df[(df['odds_away_win'] > 1.01) & (df['odds_away_win'] < 50)]
        print(f"  After ML odds filtering: {len(df)} matches")
        
        df = df.sort_values('date').reset_index(drop=True)
        print(f"✓ Final dataset: {len(df)} matches with valid odds")
        return df


class FeatureEngine:
    def create(self, df):
        print("\n" + "="*80)
        print("CREATING HISTORICAL FEATURES")
        print("="*80)
        
        features = [
            'home_xg_avg', 'away_xg_avg', 'home_xg_recent', 'away_xg_recent',
            'home_ppg_avg', 'away_ppg_avg', 'home_elo', 'away_elo', 'elo_diff',
            'h2h_total_goals', 'home_form', 'away_form', 'league_avg_goals',
            'home_goals_avg', 'away_goals_avg'
        ]
        
        for col in features:
            df[col] = np.nan
        
        team_elo = {}
        
        for i in range(len(df)):
            if i % 500 == 0:
                print(f"  [{i}/{len(df)}]")
            
            home = df.iloc[i]['home_team']
            away = df.iloc[i]['away_team']
            past = df.iloc[:i]
            
            if home not in team_elo:
                team_elo[home] = 1500
            if away not in team_elo:
                team_elo[away] = 1500
            
            df.at[i, 'home_elo'] = team_elo[home]
            df.at[i, 'away_elo'] = team_elo[away]
            df.at[i, 'elo_diff'] = team_elo[home] - team_elo[away]
            
            if len(past) > 0:
                # Home history
                home_past = past[(past['home_team'] == home) | (past['away_team'] == home)]
                if len(home_past) >= 5:
                    home_xg, home_ppg, home_pts, home_goals = [], [], [], []
                    for _, m in home_past.iterrows():
                        if m['home_team'] == home:
                            home_xg.append(m['pre_home_xg'])
                            home_ppg.append(m['home_ppg'])
                            home_goals.append(m['home_goals'])
                            home_pts.append(3 if m['home_goals'] > m['away_goals'] else (1 if m['home_goals'] == m['away_goals'] else 0))
                        else:
                            home_xg.append(m['pre_away_xg'])
                            home_ppg.append(m['away_ppg'])
                            home_goals.append(m['away_goals'])
                            home_pts.append(3 if m['away_goals'] > m['home_goals'] else (1 if m['away_goals'] == m['home_goals'] else 0))
                    
                    df.at[i, 'home_xg_avg'] = np.mean(home_xg)
                    df.at[i, 'home_xg_recent'] = np.mean(home_xg[-5:])
                    df.at[i, 'home_ppg_avg'] = np.mean(home_ppg)
                    df.at[i, 'home_form'] = sum(home_pts[-5:])
                    df.at[i, 'home_goals_avg'] = np.mean(home_goals)
                
                # Away history
                away_past = past[(past['home_team'] == away) | (past['away_team'] == away)]
                if len(away_past) >= 5:
                    away_xg, away_ppg, away_pts, away_goals = [], [], [], []
                    for _, m in away_past.iterrows():
                        if m['home_team'] == away:
                            away_xg.append(m['pre_home_xg'])
                            away_ppg.append(m['home_ppg'])
                            away_goals.append(m['home_goals'])
                            away_pts.append(3 if m['home_goals'] > m['away_goals'] else (1 if m['home_goals'] == m['away_goals'] else 0))
                        else:
                            away_xg.append(m['pre_away_xg'])
                            away_ppg.append(m['away_ppg'])
                            away_goals.append(m['away_goals'])
                            away_pts.append(3 if m['away_goals'] > m['home_goals'] else (1 if m['away_goals'] == m['home_goals'] else 0))
                    
                    df.at[i, 'away_xg_avg'] = np.mean(away_xg)
                    df.at[i, 'away_xg_recent'] = np.mean(away_xg[-5:])
                    df.at[i, 'away_ppg_avg'] = np.mean(away_ppg)
                    df.at[i, 'away_form'] = sum(away_pts[-5:])
                    df.at[i, 'away_goals_avg'] = np.mean(away_goals)
                
                # H2H
                h2h = past[((past['home_team'] == home) & (past['away_team'] == away)) |
                          ((past['home_team'] == away) & (past['away_team'] == home))]
                df.at[i, 'h2h_total_goals'] = h2h['total_goals'].mean() if len(h2h) > 0 else 2.5
                
                # League
                league_past = past[past['league'] == df.iloc[i]['league']]
                df.at[i, 'league_avg_goals'] = league_past['total_goals'].mean() if len(league_past) > 0 else 2.5
            
            # Update Elo
            result = 1.0 if df.iloc[i]['home_goals'] > df.iloc[i]['away_goals'] else (0.0 if df.iloc[i]['home_goals'] < df.iloc[i]['away_goals'] else 0.5)
            expected = 1 / (1 + 10 ** ((team_elo[away] - team_elo[home]) / 400))
            team_elo[home] += 20 * (result - expected)
            team_elo[away] += 20 * ((1 - result) - (1 - expected))
        
        df = df.dropna(subset=features)
        df = df.iloc[30:].reset_index(drop=True)
        print(f"✓ Features created: {len(df)} matches")
        return df


class ModelTrainer:
    def __init__(self):
        self.scalers = {}
        self.models = {}
    
    def train(self, df_train):
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        os.makedirs('models', exist_ok=True)
        
        # === O/U 2.5 MODEL ===
        print("\n→ O/U 2.5 Model")
        ou_features = [
            'pre_total_xg', 'CTMCL', 'pre_home_xg', 'pre_away_xg', 
            'o25_potential', 'o35_potential', 'o45_potential', 'btts_potential',
            'home_xg_avg', 'away_xg_avg', 'home_xg_recent', 'away_xg_recent',
            'home_goals_avg', 'away_goals_avg', 'league_avg_goals',
            'home_form', 'away_form', 'home_ppg_avg', 'away_ppg_avg',
            'xg_diff', 'xg_ratio', 'ppg_diff'
        ]
        ou_features = [f for f in ou_features if f in df_train.columns]
        
        y_train_ou = (df_train['total_goals'] > 2.5).astype(int)
        
        scaler_ou = StandardScaler()
        X_train_ou = scaler_ou.fit_transform(df_train[ou_features])
        
        xgb_ou = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.02, subsample=0.8, 
                               colsample_bytree=0.8, random_state=RANDOM_SEED, **GPU_PARAMS)
        xgb_ou.fit(X_train_ou, y_train_ou, verbose=False)
        
        gb_ou = GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.02, subsample=0.8, random_state=RANDOM_SEED)
        gb_ou.fit(X_train_ou, y_train_ou)
        
        rf_ou = RandomForestClassifier(n_estimators=250, max_depth=15, min_samples_leaf=2, random_state=RANDOM_SEED)
        rf_ou.fit(X_train_ou, y_train_ou)
        
        print(f"  O/U 2.5 Model trained")
        
        self.models['ou'] = {'xgb': xgb_ou, 'gb': gb_ou, 'rf': rf_ou}
        self.scalers['ou'] = scaler_ou
        
        # === MONEYLINE MODEL ===
        print("\n→ Moneyline Model")
        ml_features = [
            'xg_diff', 'ppg_diff', 'home_form', 'away_form',
            'home_xg_avg', 'away_xg_avg',
            'home_xg_recent', 'away_xg_recent', 'pre_home_xg', 'pre_away_xg',
            'home_ppg_avg', 'away_ppg_avg', 'xg_ratio',
            'league_avg_goals', 'home_goals_avg', 'away_goals_avg'
        ]
        ml_features = [f for f in ml_features if f in df_train.columns]
        
        y_train_ml = (df_train['home_goals'] > df_train['away_goals']).astype(int)
        
        scaler_ml = StandardScaler()
        X_train_ml = scaler_ml.fit_transform(df_train[ml_features])
        
        xgb_ml = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.02, subsample=0.8,
                               colsample_bytree=0.8, random_state=RANDOM_SEED, **GPU_PARAMS)
        xgb_ml.fit(X_train_ml, y_train_ml, verbose=False)
        
        gb_ml = GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.02, subsample=0.8, random_state=RANDOM_SEED)
        gb_ml.fit(X_train_ml, y_train_ml)
        
        print(f"  Moneyline Model trained")
        
        self.models['ml'] = {'xgb': xgb_ml, 'gb': gb_ml}
        self.scalers['ml'] = scaler_ml
        
        # Save separate pkl files
        joblib.dump(self.models['ou'], 'models/ou_model.pkl')
        joblib.dump(self.models['ml'], 'models/ml_model.pkl')
        joblib.dump(self.scalers['ou'], 'models/ou_scaler.pkl')
        joblib.dump(self.scalers['ml'], 'models/ml_scaler.pkl')
        print("\n✓ Models and scalers saved separately")
        
        return {
            'ou_features': ou_features,
            'ml_features': ml_features
        }


class ConfidenceCalculator:
    """Fottyy-main Confidence Method: 60% Base + 40% Margin"""
    
    @staticmethod
    def calculate(final_pred):
        """
        Calculate confidence using fottyy-main method:
        - 60% Base Confidence (from max probability)
        - 40% Margin Confidence (gap between 1st and 2nd predictions)
        
        Args:
            final_pred: Final ensemble prediction probability (e.g., 0.65 for Over 2.5)
        
        Returns:
            confidence: Confidence score (0-100)
        """
        # For binary predictions (Over/Under or Home/Away)
        # We have two probabilities that sum to 1.0
        prob_predicted = final_pred
        prob_other = 1 - final_pred
        
        # Get max probability and margin
        max_prob = max(prob_predicted, prob_other)
        margin = abs(prob_predicted - prob_other)
        
        # 1. Base confidence from absolute probability (max 60%)
        base_confidence = max_prob * 60
        
        # 2. Additional confidence from margin (max 40%)
        # Margin of 0.20 (20%) or more gets full 40%
        margin_confidence = min(margin * 200, 40)
        
        # 3. Total confidence (max 100%)
        confidence = base_confidence + margin_confidence
        
        return confidence


class Predictor:
    def run(self, filepath):
        print("\n" + "="*80)
        print("🚀 FOOTBALL PREDICTOR - FOTTYY-MAIN CONFIDENCE")
        print("60% BASE CONFIDENCE + 40% MARGIN CONFIDENCE")
        print("="*80)
        print("TRAINING ON ALL AVAILABLE DATA")
        print("="*80)
        
        loader = DataLoader()
        df = loader.load(filepath)
        
        engine = FeatureEngine()
        df = engine.create(df)
        
        print(f"\n✓ Using all {len(df)} matches for training")
        
        trainer = ModelTrainer()
        features_info = trainer.train(df)
        
        print("\n" + "="*80)
        print("✅ COMPLETE")
        print("="*80)
        
        return features_info


if __name__ == "__main__":
    predictor = Predictor()
    features_info = predictor.run('top.csv')