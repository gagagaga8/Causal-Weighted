"""
LightGBM/GBM Causal Model - Inference 
use pair Data RowRRTstart 
 : lightgbm_best.pkl, gbm_causal_best.pkl
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os


# allFeatureColumnTable withTraining 
SAFE_FEATURES = [
    'admission_age', 'gender', 'weight', 'sofa_24hours',
    'aki_stage', 'aki_stage_creat', 'aki_stage_uo', 'creat',
    'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
    'uo_k1', 'bun_k1', 'pot_k1', 'ph_k1', 'creat_k1',
    'uo_k2', 'bun_k2', 'pot_k2', 'ph_k2', 'creat_k2',
    'uo_k3', 'bun_k3', 'pot_k3', 'ph_k3', 'creat_k3'
]


def load_model(model_path='checkpoints/lightgbm_best.pkl'):
    """LoadingTraining Model"""
    print(f"LoadingModel: {model_path}")
    checkpoint = joblib.load(model_path)
    return checkpoint['model'], checkpoint['scaler'], checkpoint['features']


def predict(model, scaler, X, features):
    """PredictionRRTstart """
    # Feature
    available = [c for c in features if c in X.columns]
    if len(available) != len(features):
        missing = set(features) - set(available)
        print(f"Warning: MissingFeature {missing}")
    
    X_input = X[available].copy()
    
    # ProcessingSex
    if 'gender' in X_input.columns:
        X_input['gender'] = X_input['gender'].map({'M': 1, 'F': 0}).fillna(0)
    
    X_input = X_input.fillna(0).values
    
    # Standardize
    X_scaled = scaler.transform(X_input)
    
    # Prediction
    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)[:, 1]
    
    return pred, prob


def main():
    parser = argparse.ArgumentParser(description='GBMCausal ModelInference')
    parser.add_argument('--data_path', type=str, required=True, help='InputDataPath')
    parser.add_argument('--model_path', type=str, default='checkpoints/gbm_causal_best.pkl')
    parser.add_argument('--output_path', type=str, default='results/predictions.csv')
    args = parser.parse_args()
    
    # Load model
    model, scaler, features = load_model(args.model_path)
    
    # Load data
    print(f"\nLoad data: {args.data_path}")
    if args.data_path.endswith('.feather'):
        df = pd.read_feather(args.data_path)
    else:
        df = pd.read_csv(args.data_path)
    print(f"  Sample Size: {len(df)}")
    
    # Prediction
    pred, prob = predict(model, scaler, df, features)
    
    # SaveResults
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = pd.DataFrame({
        'recommended': pred,
        'probability': prob
    })
    
    # if has ComputationADR
    if 'a1' in df.columns:
        actual = ((df['a1'] == 1) | (df['a2'] == 1) | (df['a3'] == 1)).astype(int).values
        results['actual'] = actual
        results['correct'] = (pred == actual).astype(int)
        adr = results['correct'].mean()
        print(f"\nAgreement with actual decisions (ADR): {adr:.2%}")
    
    results.to_csv(args.output_path, index=False)
    print(f"\nResults saved: {args.output_path}")
    
    # statistics
    print(f"\n statistics:")
    print(f" startRRT: {pred.sum()} ({pred.mean():.2%})")
    print(f" notstart: {len(pred)-pred.sum()} ({1-pred.mean():.2%})")


if __name__ == '__main__':
    main()
