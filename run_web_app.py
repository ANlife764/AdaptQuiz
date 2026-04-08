# check_skills.py - Run this to see available skills

import joblib
import pandas as pd

# Load the skill encoder
skill_encoder = joblib.load('./web_app/models/skill_encoder.pkl')

# Get all skill names
all_skills = skill_encoder.classes_
print("Skills your model knows:")
for i, skill in enumerate(all_skills):
    print(f"   {i}. {skill}")

print(f"\nTotal: {len(all_skills)} skills")