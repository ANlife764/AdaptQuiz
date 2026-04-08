# save_models.py - Corrected file paths

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("=" * 60)
print("TRAINING AND SAVING GRADIENT BOOSTING MODEL")
print("=" * 60)

# Get the correct base path (project root)
# Since this script is in src/, we need to go up one level
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'cleaned_data.parquet')

print(f"\nLooking for data at: {DATA_PATH}")

# Check if file exists
if not os.path.exists(DATA_PATH):
    print(f"\n❌ ERROR: File not found!")
    print(f"   Expected location: {DATA_PATH}")
    print(f"\n   Your project structure should be:")
    print(f"   AdaptQuiz/")
    print(f"   ├── data/")
    print(f"   │   └── processed/")
    print(f"   │       └── cleaned_data.parquet  ← This file is missing!")
    print(f"   ├── src/")
    print(f"   │   └── save_models.py  ← You are here")
    print(f"   └── web_app/")
    print(f"\n   Please run Notebook 01 first to create cleaned_data.parquet")
    exit(1)

# Load processed data
print("\n✅ Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"Data shape: {df.shape}")

# Encode categorical variables
skill_encoder = LabelEncoder()
student_encoder = LabelEncoder()

df['skill_id'] = skill_encoder.fit_transform(df['skill'])
df['student_id'] = student_encoder.fit_transform(df['user_id'])

# Sort by student and sequence
df = df.sort_values(['student_id', 'sequence_id']).reset_index(drop=True)

# Create features (same as Notebook 2)
print("\nCreating features...")

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

# Streak features
def calculate_streak_past(group):
    streak = 0
    streaks = [0]
    for correct in group[:-1]:
        if correct == 1:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    return streaks

df['streak'] = df.groupby('student_id')['correct'].transform(calculate_streak_past)

# Feature columns
feature_cols = [
    'skill_id', 'prev_correct', 'prev_2_correct', 'prev_3_correct',
    'rolling_avg_5', 'rolling_avg_10', 'skill_prev_correct',
    'skill_attempt_count', 'log_response_time', 'problem_difficulty',
    'normalized_position', 'streak'
]

# Drop NaN rows
df = df.dropna(subset=feature_cols)
print(f"Data after cleaning: {df.shape}")

# Prepare features and target
X = df[feature_cols]
y = df['correct'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting
print("\nTraining Gradient Boosting...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Model Performance:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   AUC: {auc:.4f}")

# Create web_app/models directory
WEB_MODELS_PATH = os.path.join(BASE_PATH, 'web_app', 'models')
os.makedirs(WEB_MODELS_PATH, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(WEB_MODELS_PATH, 'gradient_boosting_model.pkl'))
print("\n✓ Gradient Boosting model saved")

# Save scaler
joblib.dump(scaler, os.path.join(WEB_MODELS_PATH, 'scaler.pkl'))
print("✓ Scaler saved")

# Save feature columns
with open(os.path.join(WEB_MODELS_PATH, 'feature_columns.json'), 'w') as f:
    json.dump(feature_cols, f)
print("✓ Feature columns saved")

# Save skill encoder
joblib.dump(skill_encoder, os.path.join(WEB_MODELS_PATH, 'skill_encoder.pkl'))
print("✓ Skill encoder saved")

print("\n" + "=" * 60)
print("✅ ALL MODELS SAVED SUCCESSFULLY!")
print("=" * 60)
print(f"\nFiles saved in: {WEB_MODELS_PATH}")
print("   - gradient_boosting_model.pkl")
print("   - scaler.pkl")
print("   - feature_columns.json")
print("   - skill_encoder.pkl")
print("\nNow run the web interface:")
print("   cd web_app")
print("   python app.py")