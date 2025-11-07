"""
FOOTBALL PREDICTOR - INFERENCE
Uses pre-trained models from models/ folder to generate predictions
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42

class FeaturePreprocessor:
    def create_features(self, df):
        print("\n" + "="*80)
        print("CREATING FEATURES FOR INFERENCE")
        print("="*80)
        
        df = df.copy()
        
        # Pre-match xG
        df['pre_home_xg'] = df['team_a_xg_prematch']
        df['pre_away_xg'] = df['team_b_xg_prematch']
        df['pre_total_xg'] = df['team_a_xg_prematch'] + df['team_b_xg_prematch']
        df['xg_diff'] = df['team_a_xg_prematch'] - df['team_b_xg_prematch']
        df['xg_ratio'] = df['team_a_xg_prematch'] / (df['team_b_xg_prematch'] + 0.01)
        
        # PPG
        df['ppg_diff'] = df['pre_match_home_ppg'] - df['pre_match_away_ppg']
        
        # Direct features from CSV
        df['home_xg_avg'] = df['home_xg_avg']
        df['away_xg_avg'] = df['away_xg_avg']
        df['home_xg_recent'] = df['home_goals_conceded_avg']
        df['away_xg_recent'] = df['away_goals_conceded_avg']
        
        # home_goals_avg and away_goals_avg - use if available, fallback to form_points
        if 'home_goals_avg' in df.columns:
            df['home_goals_avg'] = df['home_goals_avg']
        else:
            df['home_goals_avg'] = df['home_form_points']
            print("  ⚠ Using home_form_points for home_goals_avg")
        
        if 'away_goals_avg' in df.columns:
            df['away_goals_avg'] = df['away_goals_avg']
        else:
            df['away_goals_avg'] = df['away_form_points']
            print("  ⚠ Using away_form_points for away_goals_avg")
        
        df['home_form'] = df['home_form_points']
        df['away_form'] = df['away_form_points']
        df['home_ppg_avg'] = df['pre_match_home_ppg']
        df['away_ppg_avg'] = df['pre_match_away_ppg']
        
        print(f"✓ Features created for {len(df)} matches")
        return df


class ConfidenceCalculator:
    """Fottyy-main Confidence Method: 60% Base + 40% Margin"""
    
    @staticmethod
    def calculate(final_pred):
        prob_predicted = final_pred
        prob_other = 1 - final_pred
        max_prob = max(prob_predicted, prob_other)
        margin = abs(prob_predicted - prob_other)
        
        base_confidence = max_prob * 60
        margin_confidence = min(margin * 200, 40)
        confidence = base_confidence + margin_confidence
        
        return confidence


class Predictor:
    def __init__(self):
        self.calc = ConfidenceCalculator()
        self.load_models()
    
    def load_models(self):
        print("\n" + "="*80)
        print("LOADING MODELS AND SCALERS")
        print("="*80)
        
        self.ou_models = joblib.load('models/ou_model.pkl')
        self.ml_models = joblib.load('models/ml_model.pkl')
        self.ou_scaler = joblib.load('models/ou_scaler.pkl')
        self.ml_scaler = joblib.load('models/ml_scaler.pkl')
        
        print("✓ O/U Models loaded (XGB, GB, RF)")
        print("✓ ML Models loaded (XGB, GB)")
        print("✓ O/U Scaler loaded")
        print("✓ ML Scaler loaded")
    
    def predict(self, df):
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS")
        print("="*80)
        
        # O/U Features
        ou_features = [
            'pre_total_xg', 'CTMCL', 'pre_home_xg', 'pre_away_xg', 
            'o25_potential', 'o35_potential', 'o45_potential', 'btts_potential',
            'home_xg_avg', 'away_xg_avg', 'home_xg_recent', 'away_xg_recent',
            'home_goals_avg', 'away_goals_avg', 'league_avg_goals',
            'home_form', 'away_form', 'home_ppg_avg', 'away_ppg_avg',
            'xg_diff', 'xg_ratio', 'ppg_diff'
        ]
        ou_features = [f for f in ou_features if f in df.columns]
        
        # ML Features
        ml_features = [
            'xg_diff', 'ppg_diff', 'home_form', 'away_form',
            'home_xg_avg', 'away_xg_avg',
            'home_xg_recent', 'away_xg_recent', 'pre_home_xg', 'pre_away_xg',
            'home_ppg_avg', 'away_ppg_avg', 'xg_ratio',
            'league_avg_goals', 'home_goals_avg', 'away_goals_avg'
        ]
        ml_features = [f for f in ml_features if f in df.columns]
        
        # Prepare data
        X_ou = self.ou_scaler.transform(df[ou_features])
        X_ml = self.ml_scaler.transform(df[ml_features])
        
        # O/U Predictions
        print("\n→ O/U 2.5 Predictions")
        pred_ou_xgb = self.ou_models['xgb'].predict_proba(X_ou)[:, 1]
        pred_ou_gb = self.ou_models['gb'].predict_proba(X_ou)[:, 1]
        pred_ou_rf = self.ou_models['rf'].predict_proba(X_ou)[:, 1]
        
        market_over_prob = 1 / df['odds_ft_over25'].values
        market_under_prob = 1 / df['odds_ft_under25'].values
        market_total = market_over_prob + market_under_prob
        market_over_prob = market_over_prob / market_total
        
        pred_ou_proba = (pred_ou_xgb * 0.36 + pred_ou_gb * 0.28 + pred_ou_rf * 0.16 + market_over_prob * 0.20)
        pred_ou = (pred_ou_proba > 0.5).astype(int)
        
        print(f"✓ O/U predictions generated")
        
        # ML Predictions
        print("\n→ Moneyline Predictions")
        pred_ml_xgb = self.ml_models['xgb'].predict_proba(X_ml)[:, 1]
        pred_ml_gb = self.ml_models['gb'].predict_proba(X_ml)[:, 1]
        
        market_home_prob = 1 / df['odds_ft_1'].values
        market_away_prob = 1 / df['odds_ft_2'].values
        market_total_ml = market_home_prob + market_away_prob
        market_home_prob = market_home_prob / market_total_ml
        
        pred_ml_proba = (pred_ml_xgb * 0.40 + pred_ml_gb * 0.40 + market_home_prob * 0.20)
        pred_ml = (pred_ml_proba > 0.5).astype(int)
        
        print(f"✓ ML predictions generated")
        
        return {
            'pred_ou': pred_ou,
            'pred_ou_proba': pred_ou_proba,
            'pred_ml': pred_ml,
            'pred_ml_proba': pred_ml_proba
        }
    
    def generate_output(self, df, predictions):
        print("\n" + "="*80)
        print("GENERATING OUTPUT CSV")
        print("="*80)
        
        output = pd.DataFrame()
        
        # Basic match info
        output['match_id'] = df['match_id']
        output['home_id'] = df['home_team_id']
        output['away_id'] = df['away_team_id']
        output['league_id'] = df['league_id']
        output['date'] = df['date']
        output['league'] = df['league_name']
        output['home_team'] = df['home_team_name']
        output['away_team'] = df['away_team_name']
        
        # O/U Predictions
        output['ou_prediction'] = np.where(predictions['pred_ou'] == 1, 'Over 2.5', 'Under 2.5')
        output['ou_probability'] = predictions['pred_ou_proba']
        output['over_2_5_odds'] = df['odds_ft_over25'].values
        output['under_2_5_odds'] = df['odds_ft_under25'].values
        
        # ML Predictions
        output['ml_prediction'] = np.where(predictions['pred_ml'] == 1, 'Home Win', 'Away Win')
        output['ml_probability'] = predictions['pred_ml_proba']
        output['home_win_odds'] = df['odds_ft_1'].values
        output['away_win_odds'] = df['odds_ft_2'].values
        
        # Calculate Confidence
        ou_confidence = np.array([self.calc.calculate(prob) for prob in predictions['pred_ou_proba']])
        ml_confidence = np.array([self.calc.calculate(prob) for prob in predictions['pred_ml_proba']])
        
        output['ou_confidence'] = ou_confidence
        output['ml_confidence'] = ml_confidence
        output['ou_confidence_level'] = np.where(ou_confidence >= 80, 'HIGH', 
                                                 np.where(ou_confidence >= 50, 'MEDIUM', 'LOW'))
        output['ml_confidence_level'] = np.where(ml_confidence >= 80, 'HIGH', 
                                                 np.where(ml_confidence >= 50, 'MEDIUM', 'LOW'))
        
        output = output.reset_index(drop=True)
        print(f"✓ Output generated with {len(output)} predictions")
        
        return output


class InferencePipeline:
    def run(self, input_csv):
        print("\n" + "="*80)
        print("🚀 FOOTBALL PREDICTOR - INFERENCE")
        print("="*80)
        
        # Load data
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df)} matches")
        
        # Create features
        preprocessor = FeaturePreprocessor()
        df = preprocessor.create_features(df)
        
        # Generate predictions
        predictor = Predictor()
        predictions = predictor.predict(df)
        
        # Generate output
        output = predictor.generate_output(df, predictions)
        
        # Save
        output_file = 'predictions_output.csv'
        output.to_csv(output_file, index=False)
        print(f"\n✓ Predictions saved to: {output_file}")
        
        print("\n" + "="*80)
        print("✅ INFERENCE COMPLETE")
        print("="*80)
        
        return output


if __name__ == "__main__":
    pipeline = InferencePipeline()
    output = pipeline.run('extracted_features_complete.csv')
