# bkt_model.py
# Bayesian Knowledge Tracing Implementation
# Run this file separately: python bkt_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class BKTModel:
    """
    Bayesian Knowledge Tracing Model
    """
    def __init__(self, p_init=0.3, p_learn=0.1, p_guess=0.25, p_slip=0.1):
        self.p_init = p_init
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip
        self.skill_params = {}  # Store per-skill parameters
        
    def update_knowledge(self, prior_knowledge, correct):
        """Update knowledge probability after observation"""
        if correct == 1:
            p_obs = prior_knowledge * (1 - self.p_slip) + (1 - prior_knowledge) * self.p_guess
            if p_obs < 1e-10:
                p_obs = 1e-10
            posterior = prior_knowledge * (1 - self.p_slip) / p_obs
        else:
            p_obs = prior_knowledge * self.p_slip + (1 - prior_knowledge) * (1 - self.p_guess)
            if p_obs < 1e-10:
                p_obs = 1e-10
            posterior = prior_knowledge * self.p_slip / p_obs
        
        return np.clip(posterior, 1e-6, 1 - 1e-6)
    
    def apply_learning(self, posterior_knowledge):
        """Apply learning after attempt"""
        return posterior_knowledge + (1 - posterior_knowledge) * self.p_learn
    
    def predict_next_correct(self, current_knowledge):
        """Predict probability of next correct answer"""
        return current_knowledge * (1 - self.p_slip) + (1 - current_knowledge) * self.p_guess
    
    def fit_sequence(self, sequence):
        """Run BKT on a sequence of responses"""
        knowledge_probs = [self.p_init]
        predicted_probs = []
        
        for correct in sequence:
            pred_prob = self.predict_next_correct(knowledge_probs[-1])
            predicted_probs.append(pred_prob)
            
            posterior = self.update_knowledge(knowledge_probs[-1], correct)
            new_knowledge = self.apply_learning(posterior)
            knowledge_probs.append(new_knowledge)
            
        return knowledge_probs[1:], predicted_probs
    
    def fit_skill(self, skill_sequences):
        """Optimize parameters for a specific skill using grid search"""
        best_params = None
        best_log_lik = -np.inf
        
        # Grid search over parameter ranges
        param_grid = {
            'p_init': [0.2, 0.3, 0.4, 0.5],
            'p_learn': [0.05, 0.1, 0.15, 0.2],
            'p_guess': [0.1, 0.2, 0.25, 0.3],
            'p_slip': [0.05, 0.1, 0.15, 0.2]
        }
        
        for p_init in param_grid['p_init']:
            for p_learn in param_grid['p_learn']:
                for p_guess in param_grid['p_guess']:
                    for p_slip in param_grid['p_slip']:
                        model = BKTModel(p_init, p_learn, p_guess, p_slip)
                        
                        total_log_lik = 0
                        for seq in skill_sequences:
                            _, preds = model.fit_sequence(seq)
                            preds = np.clip(preds, 1e-10, 1-1e-10)
                            log_lik = sum(np.log(preds[i] if correct == 1 else 1-preds[i]) 
                                         for i, correct in enumerate(seq))
                            total_log_lik += log_lik
                        
                        if total_log_lik > best_log_lik:
                            best_log_lik = total_log_lik
                            best_params = (p_init, p_learn, p_guess, p_slip)
        
        self.p_init, self.p_learn, self.p_guess, self.p_slip = best_params
        return self.get_parameters()
    
    def get_parameters(self):
        """Return model parameters"""
        return {
            'p_init': self.p_init,
            'p_learn': self.p_learn,
            'p_guess': self.p_guess,
            'p_slip': self.p_slip
        }

def prepare_data_for_bkt(df, min_skill_sequences=10, min_sequence_length=3):
    """Prepare student-skill sequences for BKT"""
    print("Preparing data for BKT...")
    
    # Encode skills
    skill_encoder = LabelEncoder()
    df['skill_id'] = skill_encoder.fit_transform(df['skill'])
    
    # Create sequences per skill
    skill_sequences = {}
    
    for skill_id in df['skill_id'].unique():
        skill_data = df[df['skill_id'] == skill_id]
        sequences = []
        
        for student_id in skill_data['student_id'].unique():
            student_data = skill_data[skill_data['student_id'] == student_id].sort_values('sequence_id')
            sequence = student_data['correct'].tolist()
            
            if len(sequence) >= min_sequence_length:
                sequences.append(sequence)
        
        if len(sequences) >= min_skill_sequences:
            skill_sequences[skill_id] = sequences
    
    return skill_sequences, skill_encoder

def train_bkt_model(df):
    """Train BKT model on the dataset"""
    print("\n" + "="*60)
    print("TRAINING BKT MODEL")
    print("="*60)
    
    # Prepare data
    skill_sequences, skill_encoder = prepare_data_for_bkt(df)
    print(f"Found {len(skill_sequences)} skills with sufficient data")
    
    # Train per-skill models
    bkt_models = {}
    mastery_results = []
    
    for skill_id, sequences in skill_sequences.items():
        print(f"\nTraining BKT for skill {skill_id} ({len(sequences)} sequences)...")
        
        # Split into train and validation
        train_seqs, val_seqs = train_test_split(sequences, test_size=0.2, random_state=42)
        
        # Train model
        model = BKTModel()
        params = model.fit_skill(train_seqs)
        
        # Evaluate
        all_preds = []
        all_actuals = []
        for seq in val_seqs:
            _, preds = model.fit_sequence(seq)
            all_preds.extend(preds)
            all_actuals.extend(seq)
        
        accuracy = accuracy_score(all_actuals, np.array(all_preds) > 0.5)
        print(f"   Parameters: p_init={params['p_init']:.3f}, p_learn={params['p_learn']:.3f}, "
              f"p_guess={params['p_guess']:.3f}, p_slip={params['p_slip']:.3f}")
        print(f"   Validation Accuracy: {accuracy:.4f}")
        
        bkt_models[skill_id] = model
        
        # Compute mastery for all students
        for seq in sequences:
            knowledge_probs, _ = model.fit_sequence(seq)
            mastery_results.append({
                'skill_id': skill_id,
                'skill': skill_encoder.inverse_transform([skill_id])[0],
                'sequence_length': len(seq),
                'final_mastery': knowledge_probs[-1],
                'initial_mastery': knowledge_probs[0]
            })
    
    mastery_df = pd.DataFrame(mastery_results)
    return bkt_models, mastery_df

def main():
    """Main function to run BKT"""
    print("Loading data...")
    df = pd.read_parquet('../data/processed/cleaned_data.parquet')
    print(f"Data shape: {df.shape}")
    
    # Add student_id encoding
    student_encoder = LabelEncoder()
    df['student_id'] = student_encoder.fit_transform(df['user_id'])
    
    # Train BKT
    bkt_models, mastery_df = train_bkt_model(df)
    
    # Save results
    mastery_df.to_parquet('../data/processed/bkt_mastery.parquet', index=False)
    print(f"\n✓ BKT mastery saved to: ../data/processed/bkt_mastery.parquet")
    print(f"\nFinal BKT Results:")
    print(f"   Skills modeled: {len(bkt_models)}")
    print(f"   Student-skill pairs: {len(mastery_df)}")
    print(f"   Mean final mastery: {mastery_df['final_mastery'].mean():.4f}")
    
    return bkt_models, mastery_df

if __name__ == "__main__":
    bkt_models, mastery_df = main()