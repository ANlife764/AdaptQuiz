// web_app/static/script.js

let currentQuestion = null;
let canAnswer = true;
let questionStartTime = null;

// DOM Elements
const startScreen = document.getElementById('start-screen');
const quizScreen = document.getElementById('quiz-screen');
const resultsScreen = document.getElementById('results-screen');
const startBtn = document.getElementById('start-btn');
const nextBtn = document.getElementById('next-btn');
const restartBtn = document.getElementById('restart-btn');
const questionText = document.getElementById('question-text');
const optionsContainer = document.getElementById('options');
const feedback = document.getElementById('feedback');
const skillBadge = document.getElementById('skill-badge');
const masteryFill = document.getElementById('mastery-fill');
const masteryValue = document.getElementById('mastery-value');
const scoreSpan = document.getElementById('score');
const streakSpan = document.getElementById('streak');
const qCountSpan = document.getElementById('q-count');
const accuracySpan = document.getElementById('accuracy');
const skillProgress = document.getElementById('skill-progress');

// Start Quiz
startBtn.addEventListener('click', async () => {
    const studentName = document.getElementById('student-name').value;
    
    const response = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: studentName || 'student_' + Date.now() })
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
        showScreen('quiz');
        resetUI();
        loadNextQuestion();
        
        // Show weak skills
        if (data.weak_skills && data.weak_skills.length > 0) {
            showTemporaryMessage(`🎯 Focus areas: ${data.weak_skills.join(', ')}`, 3000);
        }
    }
});

// Load Next Question
async function loadNextQuestion() {
    canAnswer = true;
    nextBtn.style.display = 'none';
    feedback.innerHTML = '';
    feedback.className = '';
    
    // Reset option buttons
    const buttons = document.querySelectorAll('.option-btn');
    buttons.forEach(btn => {
        btn.disabled = false;
        btn.classList.remove('correct-highlight', 'wrong-highlight');
    });
    
    try {
        const response = await fetch('/api/question');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }
        
        currentQuestion = data;
        questionStartTime = Date.now();
        
        // Update UI
        questionText.textContent = data.question;
        skillBadge.textContent = `📚 ${data.skill}`;
        
        const masteryPercent = Math.round(data.mastery * 100);
        masteryFill.style.width = `${masteryPercent}%`;
        masteryValue.textContent = `${masteryPercent}%`;
        
        // Update options
        optionsContainer.innerHTML = '';
        data.options.forEach(option => {
            const btn = document.createElement('button');
            btn.className = 'option-btn';
            btn.textContent = option;
            btn.onclick = () => submitAnswer(option);
            optionsContainer.appendChild(btn);
        });
        
    } catch (error) {
        console.error('Error loading question:', error);
    }
}

// Submit Answer
async function submitAnswer(answer) {
    if (!canAnswer) return;
    canAnswer = false;
    
    const responseTime = (Date.now() - questionStartTime) / 1000;
    
    // Disable all option buttons
    const buttons = document.querySelectorAll('.option-btn');
    buttons.forEach(btn => btn.disabled = true);
    
    const response = await fetch('/api/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer, response_time: responseTime })
    });
    
    const result = await response.json();
    
    // Highlight correct/wrong answers
    buttons.forEach(btn => {
        if (btn.textContent === result.correct_answer) {
            btn.classList.add('correct-highlight');
        }
        if (btn.textContent === answer && !result.correct) {
            btn.classList.add('wrong-highlight');
        }
    });
    
    // Show feedback
    if (result.correct) {
        let message = `✅ Correct! +${result.points_earned} points`;
        if (result.streak_bonus > 0) {
            message += ` 🔥 Streak bonus! +${result.streak_bonus}`;
        }
        feedback.innerHTML = message;
        feedback.className = 'feedback correct';
    } else {
        feedback.innerHTML = `❌ Incorrect! The correct answer was: ${result.correct_answer}`;
        feedback.className = 'feedback wrong';
    }
    
    // Update stats
    scoreSpan.textContent = result.total_score;
    streakSpan.textContent = result.streak;
    qCountSpan.textContent = result.questions_answered;
    
    // Update mastery display
    if (result.mastery_update) {
        const skill = Object.keys(result.mastery_update)[0];
        const newMastery = result.mastery_update[skill];
        masteryFill.style.width = `${Math.round(newMastery * 100)}%`;
        masteryValue.textContent = `${Math.round(newMastery * 100)}%`;
    }
    
    // Update progress
    updateProgress();
    
    // Show next button
    nextBtn.style.display = 'block';
    
    // Check if quiz should end (after 10 questions)
    if (result.questions_answered >= 10) {
        endQuiz();
    }
}

// Update Progress Display
async function updateProgress() {
    const response = await fetch('/api/progress');
    const data = await response.json();
    
    // Update accuracy
    const accuracyPercent = Math.round(data.accuracy * 100);
    accuracySpan.textContent = `${accuracyPercent}%`;
    
    // Update skill progress
    if (data.skill_performance && Object.keys(data.skill_performance).length > 0) {
        skillProgress.innerHTML = '';
        Object.entries(data.skill_performance).forEach(([skill, perf]) => {
            const accuracy = Math.round(perf.accuracy * 100);
            const skillDiv = document.createElement('div');
            skillDiv.className = 'skill-item';
            skillDiv.innerHTML = `
                <span class="skill-name">${skill}</span>
                <div class="skill-bar-container">
                    <div class="skill-bar" style="width: ${accuracy}%"></div>
                </div>
                <span class="skill-accuracy">${accuracy}%</span>
                <span>(${perf.correct}/${perf.total})</span>
            `;
            skillProgress.appendChild(skillDiv);
        });
    }
    
    // Show weak skills remaining
    if (data.weak_skills_remaining && data.weak_skills_remaining.length > 0) {
        const weakDiv = document.createElement('div');
        weakDiv.style.marginTop = '15px';
        weakDiv.style.padding = '10px';
        weakDiv.style.background = '#fff3cd';
        weakDiv.style.borderRadius = '10px';
        weakDiv.innerHTML = `🎯 Still weak: ${data.weak_skills_remaining.slice(0, 3).join(', ')}`;
        skillProgress.appendChild(weakDiv);
    }
}

// End Quiz
async function endQuiz() {
    const response = await fetch('/api/progress');
    const data = await response.json();
    
    const resultsSummary = document.getElementById('results-summary');
    resultsSummary.innerHTML = `
        <div class="result-card">
            <h4>📊 Final Statistics</h4>
            <p>Questions Answered: ${data.questions_answered}</p>
            <p>Correct Answers: ${data.correct_count}</p>
            <p>Final Accuracy: ${Math.round(data.accuracy * 100)}%</p>
            <p>Total Score: ${data.total_score}</p>
            <p>Best Streak: ${data.current_streak}</p>
        </div>
        <div class="result-card">
            <h4>🎯 Skills Improved</h4>
            <p>You practiced ${Object.keys(data.skill_performance).length} different skills</p>
            <p>Check your progress above to see improvement areas!</p>
        </div>
    `;
    
    showScreen('results');
}

// Reset UI
function resetUI() {
    scoreSpan.textContent = '0';
    streakSpan.textContent = '0';
    qCountSpan.textContent = '0';
    accuracySpan.textContent = '0%';
    skillProgress.innerHTML = '<p>Answer questions to see your skill progress...</p>';
}

// Show Temporary Message
function showTemporaryMessage(message, duration) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'temporary-message';
    msgDiv.textContent = message;
    msgDiv.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: #28a745;
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        z-index: 1000;
        animation: fadeOut ${duration}ms ease;
    `;
    document.body.appendChild(msgDiv);
    setTimeout(() => msgDiv.remove(), duration);
}

// Screen Navigation
function showScreen(screenName) {
    startScreen.classList.remove('active');
    quizScreen.classList.remove('active');
    resultsScreen.classList.remove('active');
    
    if (screenName === 'start') startScreen.classList.add('active');
    if (screenName === 'quiz') quizScreen.classList.add('active');
    if (screenName === 'results') resultsScreen.classList.add('active');
}

// Next Question
nextBtn.addEventListener('click', () => {
    loadNextQuestion();
});

// Restart Quiz
restartBtn.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    showScreen('start');
    document.getElementById('student-name').value = '';
});

// Add CSS animation for temporary message
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        0% { opacity: 1; transform: translateX(-50%) translateY(0); }
        70% { opacity: 1; }
        100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
    }
`;
document.head.appendChild(style);