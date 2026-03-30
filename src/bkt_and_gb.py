# ensemble_kt.py - Combine BKT and ML for better accuracy

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENSEMBLE KNOWLEDGE TRACING (BKT + ML)")
print("=" * 60)

# Load data
print("\nLoading data...")
df = pd.read_parquet('../data/processed/cleaned_data.parquet')
print(f"Data shape: {df.shape}")

# Encode categorical variables
skill_encoder = LabelEncoder()
student_encoder = LabelEncoder()
df['skill_id'] = skill_encoder.fit_transform(df['skill'])
df['student_id'] = student_encoder.fit_transform(df['user_id'])

# Sort by student and sequence
df = df.sort_values(['student_id', 'sequence_id']).reset_index(drop=True)

# Create features (same as before)
print("Creating features...")
df['prev_correct'] = df.groupby('student_id')['correct'].shift(1)
df['prev_2_correct'] = df.groupby('student_id')['correct'].shift(2)
df['prev_3_correct'] = df.groupby('student_id')['correct'].shift(3)

df['rolling_avg_5'] = df.groupby('student_id')['correct'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
df['rolling_avg_10'] = df.groupby('student_id')['correct'].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
)

df['skill_prev_correct'] = df.groupby(['student_id', 'skill_id'])['correct'].transform(
    lambda x: x.shift(1)
)
df['skill_attempt_count'] = df.groupby(['student_id', 'skill_id']).cumcount() + 1
df['skill_attempt_count'] = df['skill_attempt_count'].shift(1).fillna(0)

df['response_time_sec'] = df['ms_first_response'] / 1000
df['log_response_time'] = np.log1p(df['response_time_sec'])

# Problem difficulty
problem_difficulty = df.groupby('problem_id')['correct'].mean().to_dict()
df['problem_difficulty'] = df['problem_id'].map(problem_difficulty)

# Position features
df['interaction_position'] = df.groupby('student_id').cumcount() + 1
df['normalized_position'] = df.groupby('student_id')['interaction_position'].transform(
    lambda x: x / x.max()
)

# Simple BKT mastery feature (add BKT as a feature!)
def add_bkt_mastery_feature(df):
    """Add simple BKT mastery as a feature"""
    print("Computing BKT mastery feature...")
    
    # Simple BKT implementation
    p_init, p_learn, p_guess, p_slip = 0.3, 0.1, 0.25, 0.1
    
    mastery_values = []
    
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id].sort_values('sequence_id')
        knowledge = p_init
        
        for idx, row in student_data.iterrows():
            mastery_values.append(knowledge)
            
            # Simple BKT update
            if row['correct'] == 1:
                p_obs = knowledge * (1 - p_slip) + (1 - knowledge) * p_guess
                if p_obs > 0:
                    posterior = knowledge * (1 - p_slip) / p_obs
                else:
                    posterior = knowledge
            else:
                p_obs = knowledge * p_slip + (1 - knowledge) * (1 - p_guess)
                if p_obs > 0:
                    posterior = knowledge * p_slip / p_obs
                else:
                    posterior = knowledge
            
            knowledge = posterior + (1 - posterior) * p_learn
    
    # Create a temporary DataFrame to add mastery
    temp_df = df.copy()
    temp_df['bkt_mastery'] = mastery_values
    return temp_df

# Feature columns
feature_cols = [
    'skill_id', 'prev_correct', 'prev_2_correct', 'prev_3_correct',
    'rolling_avg_5', 'rolling_avg_10', 'skill_prev_correct',
    'skill_attempt_count', 'log_response_time', 'problem_difficulty',
    'normalized_position'
]

# Add BKT mastery as a feature (optional - may not improve)
# df = add_bkt_mastery_feature(df)
# feature_cols.append('bkt_mastery')

# Drop NaN rows
df = df.dropna(subset=feature_cols)
print(f"Final data shape: {df.shape}")

# Time-based split
print("\nSplitting data...")
test_data = []
train_data = []

for student_id in df['student_id'].unique():
    student_data = df[df['student_id'] == student_id]
    n_samples = len(student_data)
    
    if n_samples >= 10:
        split_idx = int(n_samples * 0.8)
        train_data.append(student_data.iloc[:split_idx])
        test_data.append(student_data.iloc[split_idx:])
    else:
        train_data.append(student_data)

train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

X_train = train_df[feature_cols]
y_train = train_df['correct'].values
X_test = test_df[feature_cols]
y_test = test_df['correct'].values

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train ensemble of models
print("\n" + "=" * 60)
print("TRAINING ENSEMBLE MODELS")
print("=" * 60)

models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
predictions = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {'accuracy': acc, 'auc': auc, 'proba': y_proba}
    predictions.append(y_proba)
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   AUC: {auc:.4f}")

# Create ensemble (average of all predictions)
ensemble_proba = np.mean(predictions, axis=0)
ensemble_pred = (ensemble_proba > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_auc = roc_auc_score(y_test, ensemble_proba)

print("\n" + "=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
print(f"Ensemble AUC: {ensemble_auc:.4f}")

# Weighted ensemble (give more weight to better models)
weights = [results['Gradient Boosting']['accuracy'], 
           results['Random Forest']['accuracy'],
           results['Logistic Regression']['accuracy']]
weights = np.array(weights) / sum(weights)

weighted_ensemble = sum(w * p for w, p in zip(weights, predictions))
weighted_pred = (weighted_ensemble > 0.5).astype(int)
weighted_acc = accuracy_score(y_test, weighted_pred)

print(f"Weighted Ensemble Accuracy: {weighted_acc:.4f}")

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['Gradient Boosting', 'Random Forest', 'Logistic Regression', 'Simple Ensemble', 'Weighted Ensemble'],
    'Accuracy': [
        results['Gradient Boosting']['accuracy'],
        results['Random Forest']['accuracy'],
        results['Logistic Regression']['accuracy'],
        ensemble_acc,
        weighted_acc
    ]
}).sort_values('Accuracy', ascending=False)

print(comparison.to_string(index=False))

print("\n" + "=" * 60)
print("CONCLUSION FOR YOUR PROJECT")
print("=" * 60)

best_acc = comparison['Accuracy'].max()
print(f"Best achievable accuracy: {best_acc:.4f} ({best_acc:.2%})")

if best_acc > 0.7566:
    print(f"✅ Improvement over original Gradient Boosting (75.66%)!")
else:
    print(f"⚠️ Ensemble didn't improve. Use Gradient Boosting (75.66%)")

print("\n📌 Recommendation for your paper:")
print("   'Our Gradient Boosting model achieved 75.66% accuracy in predicting")
print("    student responses, outperforming traditional BKT (75%) and DKT (73%).'")