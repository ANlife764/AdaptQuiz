# web_app/app.py - Fixed version with question tracking

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import json
import os
import random
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'adapt-quiz-secret-key-2024'
app.config['SESSION_COOKIE_SIZE'] = 4096

# Load your trained Gradient Boosting model
MODEL_PATH = 'models/gradient_boosting_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURE_COLS_PATH = 'models/feature_columns.json'
SKILL_ENCODER_PATH = 'models/skill_encoder.pkl'
MASTERY_PATH = '../data/processed/student_skill_mastery.parquet'

# Global variables
model = None
scaler = None
feature_cols = None
skill_encoder = None
mastery_df = None
ALL_SKILLS = []

def load_models():
    """Load your trained Gradient Boosting model from Notebook 2"""
    global model, scaler, feature_cols, skill_encoder, mastery_df, ALL_SKILLS
    
    print("=" * 50)
    print("Loading Adapt-Quiz Models...")
    print("=" * 50)
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✓ Gradient Boosting model loaded (75.66% accuracy)")
    else:
        print("⚠️ Gradient Boosting model not found!")
        model = None
    
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✓ Scaler loaded")
    else:
        print("⚠️ Scaler not found")
        scaler = None
    
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, 'r') as f:
            feature_cols = json.load(f)
        print(f"✓ Feature columns loaded ({len(feature_cols)} features)")
    else:
        print("⚠️ Feature columns not found")
        feature_cols = None
    
    if os.path.exists(SKILL_ENCODER_PATH):
        skill_encoder = joblib.load(SKILL_ENCODER_PATH)
        ALL_SKILLS = list(skill_encoder.classes_)
        print(f"✓ Skill encoder loaded ({len(ALL_SKILLS)} skills available)")
    else:
        print("⚠️ Skill encoder not found")
        skill_encoder = None
        ALL_SKILLS = []
    
    if os.path.exists(MASTERY_PATH):
        mastery_df = pd.read_parquet(MASTERY_PATH)
        print(f"✓ Mastery data loaded ({len(mastery_df)} student-skill pairs)")
    else:
        print("⚠️ Mastery data not found")
        mastery_df = None
    
    print("=" * 50)

def load_questions():
    """Load questions from JSON file"""
    questions_path = 'data/questions.json'
    
    if os.path.exists(questions_path):
        with open(questions_path, 'r') as f:
            all_questions = json.load(f)
        
        if skill_encoder is not None:
            filtered_questions = {}
            for skill in all_questions:
                if skill in ALL_SKILLS:
                    filtered_questions[skill] = all_questions[skill]
            print(f"✓ Loaded {len(filtered_questions)} skills with questions")
            return filtered_questions
        else:
            return all_questions
    else:
        print("⚠️ questions.json not found!")
        return {}

# Load questions
QUESTIONS = load_questions()
ALL_SKILLS = [s for s in ALL_SKILLS if s in QUESTIONS]
print(f"📚 Available skills for quiz: {len(ALL_SKILLS)}")

# Precompute total questions available
TOTAL_QUESTIONS = sum(len(q) for q in QUESTIONS.values())
print(f"📚 Total questions available: {TOTAL_QUESTIONS}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_quiz():
    """Initialize quiz session"""
    data = request.json
    student_id = data.get('student_id', 'student_' + str(random.randint(1, 10000)))
    
    # Initialize session
    session['student_id'] = student_id
    session['score'] = 0
    session['questions_answered'] = 0
    session['streak'] = 0
    session['history'] = []
    session['skills_attempted'] = []
    session['asked_questions'] = []  # Track which questions have been asked
    
    # Get initial mastery
    if mastery_df is not None:
        student_data = mastery_df[mastery_df['student_id'].astype(str) == str(student_id)]
        if len(student_data) > 0:
            mastery_list = []
            for _, row in student_data.iterrows():
                skill = row['skill']
                if skill in ALL_SKILLS:
                    mastery_list.append([skill, float(row['mastery_probability'])])
            session['mastery'] = mastery_list[:30]
        else:
            mastery_list = [[skill, random.uniform(0.3, 0.7)] for skill in ALL_SKILLS[:30]]
            session['mastery'] = mastery_list
    else:
        mastery_list = [[skill, random.uniform(0.3, 0.7)] for skill in ALL_SKILLS[:30]]
        session['mastery'] = mastery_list
    
    weak_skills = [skill for skill, mastery in session['mastery'] if mastery < 0.5]
    
    print(f"Quiz started for {student_id}")
    print(f"Weak skills identified: {len(weak_skills)}")
    print(f"Total questions available: {TOTAL_QUESTIONS}")
    
    return jsonify({
        'status': 'success',
        'student_id': student_id,
        'weak_skills': weak_skills[:5],
        'total_weak': len(weak_skills),
        'mastery': {skill: round(m, 2) for skill, m in session['mastery'][:5]}
    })

@app.route('/api/question', methods=['GET'])
def get_question():
    """Get next adaptive question - never repeat same question"""
    if 'mastery' not in session:
        return jsonify({'error': 'Quiz not started'}), 400
    
    mastery_dict = {skill: m for skill, m in session['mastery']}
    skills_attempted = session.get('skills_attempted', [])
    asked_questions = session.get('asked_questions', [])
    
    # Get skills that have UNASKED questions
    available_skills = []
    for skill in ALL_SKILLS:
        if skill in QUESTIONS:
            # Check if this skill has any unasked questions
            skill_question_ids = [q['id'] for q in QUESTIONS[skill]]
            unasked = [qid for qid in skill_question_ids if qid not in asked_questions]
            if unasked:
                available_skills.append(skill)
    
    if not available_skills:
        # All questions asked! Reset asked questions
        session['asked_questions'] = []
        asked_questions = []
        available_skills = ALL_SKILLS
    
    # Adaptive selection: pick weakest skill from available skills
    weak_skills = [(skill, mastery_dict.get(skill, 0.5)) for skill in available_skills 
                   if mastery_dict.get(skill, 0.5) < 0.5]
    weak_skills.sort(key=lambda x: x[1])
    
    selected_skill = None
    if weak_skills:
        for skill, _ in weak_skills:
            if skill not in skills_attempted[-3:]:
                selected_skill = skill
                break
        if selected_skill is None:
            selected_skill = weak_skills[0][0]
    else:
        # No weak skills, pick random from available
        selected_skill = random.choice(available_skills)
    
    # Get an UNASKED question for this skill
    skill_questions = QUESTIONS.get(selected_skill, [])
    unasked_questions = [q for q in skill_questions if q['id'] not in asked_questions]
    
    if not unasked_questions:
        # No unasked questions for this skill, reset asked questions
        session['asked_questions'] = []
        asked_questions = []
        unasked_questions = skill_questions
    
    question = random.choice(unasked_questions)
    
    # Mark this question as asked
    asked_questions.append(question['id'])
    session['asked_questions'] = asked_questions[-50:]  # Keep last 50
    
    session['current_question'] = {
        'id': question['id'],
        'skill': selected_skill,
        'correct_answer': question['correct'],
        'difficulty': question['difficulty']
    }
    
    return jsonify({
        'question_id': question['id'],
        'skill': selected_skill,
        'question': question['question'],
        'options': question['options'],
        'mastery': mastery_dict.get(selected_skill, 0.5),
        'questions_remaining': TOTAL_QUESTIONS - len(asked_questions)
    })

@app.route('/api/answer', methods=['POST'])
def submit_answer():
    """Submit answer and update mastery"""
    data = request.json
    user_answer = data.get('answer')
    response_time = data.get('response_time', 5)
    
    current_q = session.get('current_question')
    if not current_q:
        return jsonify({'error': 'No active question'}), 400
    
    is_correct = (user_answer == current_q['correct_answer'])
    skill = current_q['skill']
    
    # Update score and streak
    points = 10
    streak_bonus = 0
    
    if is_correct:
        session['streak'] = session.get('streak', 0) + 1
        if session['streak'] >= 3:
            streak_bonus = 5
            points += streak_bonus
        session['score'] = session.get('score', 0) + points
    else:
        session['streak'] = 0
    
    # Update mastery
    mastery_dict = {skill: m for skill, m in session['mastery']}
    current_mastery = mastery_dict.get(skill, 0.5)
    
    if is_correct:
        new_mastery = min(1.0, current_mastery + 0.08)
    else:
        new_mastery = max(0.0, current_mastery - 0.05)
    
    mastery_dict[skill] = new_mastery
    session['mastery'] = [[s, m] for s, m in mastery_dict.items()]
    
    # Record history
    history_entry = {
        'question_id': current_q['id'],
        'skill': skill,
        'correct': is_correct,
        'points': points,
        'streak_bonus': streak_bonus,
        'response_time': response_time,
        'streak': session['streak']
    }
    history = session.get('history', [])
    history.append(history_entry)
    session['history'] = history[-20:]
    
    session['questions_answered'] = session.get('questions_answered', 0) + 1
    skills_attempted = session.get('skills_attempted', [])
    skills_attempted.append(skill)
    session['skills_attempted'] = skills_attempted[-10:]
    
    return jsonify({
        'correct': is_correct,
        'correct_answer': current_q['correct_answer'],
        'points_earned': points,
        'streak_bonus': streak_bonus,
        'total_score': session['score'],
        'streak': session['streak'],
        'questions_answered': session['questions_answered'],
        'mastery_update': {skill: round(new_mastery, 3)}
    })

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get current quiz progress"""
    history = session.get('history', [])
    correct_count = sum(1 for h in history if h['correct'])
    total = len(history)
    
    skill_performance = {}
    for h in history:
        skill = h['skill']
        if skill not in skill_performance:
            skill_performance[skill] = {'correct': 0, 'total': 0}
        skill_performance[skill]['total'] += 1
        if h['correct']:
            skill_performance[skill]['correct'] += 1
    
    for skill in skill_performance:
        skill_performance[skill]['accuracy'] = skill_performance[skill]['correct'] / skill_performance[skill]['total']
    
    mastery_dict = {skill: m for skill, m in session.get('mastery', [])}
    weak_skills = [skill for skill, m in mastery_dict.items() if m < 0.5][:5]
    
    asked_count = len(session.get('asked_questions', []))
    
    return jsonify({
        'questions_answered': total,
        'correct_count': correct_count,
        'accuracy': correct_count / total if total > 0 else 0,
        'total_score': session.get('score', 0),
        'current_streak': session.get('streak', 0),
        'skill_performance': skill_performance,
        'weak_skills_remaining': weak_skills,
        'questions_remaining': TOTAL_QUESTIONS - asked_count
    })

@app.route('/api/reset', methods=['POST'])
def reset_quiz():
    """Reset the quiz session"""
    session.clear()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    load_models()
    print("\n" + "=" * 50)
    print("🎯 ADAPT-QUIZ WITH GRADIENT BOOSTING")
    print("=" * 50)
    print(f"📍 Open in browser: http://localhost:5000")
    print(f"🤖 Model: Gradient Boosting (75.66% accuracy)")
    print(f"📚 Skills available: {len(ALL_SKILLS)}")
    print(f"📚 Total questions: {TOTAL_QUESTIONS}")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)