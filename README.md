# AdaptQuiz

AdaptQuiz is an adaptive learning quiz application that uses knowledge tracing to identify a student's weak skills and serve them targeted practice questions in real time. It combines classic Bayesian Knowledge Tracing (BKT) with a Gradient Boosting classifier trained on historical student response data, and exposes the result as a Flask web app.

## How it works

1. Historical student interaction data (skill, correctness, response time, sequence position, etc.) is cleaned and feature-engineered in the `notebooks/` pipeline.
2. Three knowledge-tracing approaches are compared:
   - **BKT** (`src/bkt.py`) – classic Bayesian Knowledge Tracing per skill.
   - **DKT** (`src/dkt.py`) – a deep knowledge tracing baseline.
   - **BKT + Gradient Boosting ensemble** (`src/bkt_and_gb.py`) – combines BKT mastery estimates with a Gradient Boosting classifier for next-response prediction.
3. The best-performing model (Gradient Boosting, ~75.7% accuracy / 0.79 AUC) is trained and serialized via `src/save_models.py`.
4. The Flask app (`web_app/app.py`) loads the trained model, scaler, skill encoder, and per-student mastery table, then adaptively selects the next question by targeting the student's weakest skills and updating mastery estimates after every answer.

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 0.628 | 0.746 |
| Random Forest | 0.756 | 0.789 |
| **Gradient Boosting** | **0.757** | **0.790** |

## Project structure

```
AdaptQuiz/
├── run_web_app.py            # Utility script to inspect available skills
├── requirements.txt
├── render.yaml                # Render.com deployment config
├── src/                        # Knowledge tracing models & training scripts
│   ├── bkt.py
│   ├── dkt.py
│   ├── bkt_and_gb.py
│   └── save_models.py
├── notebooks/                  # Exploration, modeling, and adaptive quiz notebooks
│   ├── 01_explore_data.ipynb
│   ├── 02_diff_models.ipynb
│   └── 03_adaptive_quiz.ipynb
├── data/
│   ├── processed/               # Cleaned data, mastery tables (parquet)
│   ├── metadata/                # Skill list, model comparison, feature importance
│   └── adaptive_results/        # Strategy comparison & sample quiz history
├── results/figures/            # Generated analysis charts
└── web_app/
    ├── app.py                   # Flask application & adaptive quiz API
    ├── models/                  # Trained model, scaler, skill encoder, feature columns
    ├── data/questions.json      # Question bank
    ├── templates/index.html
    └── static/ (style.css, script.js)
```

## Setup

```bash
git clone <repo-url>
cd AdaptQuiz
pip install -r requirements.txt
```

Requirements: `flask`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `gunicorn`, `pyarrow`.

## Running the app locally

```bash
cd web_app
python app.py
```

Then open `http://localhost:5000` in your browser. The console will confirm that the model, scaler, skill encoder, and mastery data have loaded correctly.

To check which skills the trained model recognizes, run:

```bash
python run_web_app.py
```

## Deployment

The project includes a `render.yaml` for one-click deployment on [Render](https://render.com):

```yaml
buildCommand: pip install -r requirements.txt
startCommand: gunicorn web_app.app:app --bind 0.0.0.0:$PORT
```

## API overview

The Flask backend exposes a small JSON API used by the front end:

- `POST /api/start` – start a new quiz session for a student, returns initial mastery estimates and weak skills.
- `GET /api/question` – fetch the next adaptively-selected question (prioritizes weak skills, avoids repeats).
- `POST /api/answer` – submit an answer; updates score, streak, and per-skill mastery.
- `GET /api/progress` – return current accuracy, score, streak, and per-skill performance breakdown.
- `POST /api/reset` – reset the quiz session.

## Retraining the model

To regenerate the trained Gradient Boosting model, scaler, and encoders from the processed data:

```bash
python src/save_models.py
```

This reads `data/processed/cleaned_data.parquet` and writes the model artifacts to `web_app/models/`.

## Notebooks

- `01_explore_data.ipynb` – exploratory data analysis on the raw student response dataset.
- `02_diff_models.ipynb` – trains and compares Logistic Regression, Random Forest, and Gradient Boosting models.
- `03_adaptive_quiz.ipynb` – simulates and evaluates adaptive question-selection strategies.

## License

Add a license of your choice here.
