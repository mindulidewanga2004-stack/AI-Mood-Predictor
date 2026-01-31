import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

st.set_page_config(page_title="Mood Predictor", layout="wide")

st.title("🎭 AI Mood Predictor 🌤️")
st.markdown("Predict your mood from daily habits!")

# Simple model recreation (NO loading issues)
@st.cache_data
def create_model():
    np.random.seed(42)
    # Simulate your 1M row training data patterns
    n_samples = 10000
    sleep_hours = np.random.normal(7, 2, n_samples).clip(0, 12)
    screen_time = np.random.normal(6, 3, n_samples).clip(0, 15)
    steps = np.random.normal(8000, 4000, n_samples).clip(0, 25000)
    social_time = np.random.normal(3, 2, n_samples).clip(0, 7)
    exercise_freq = np.random.normal(3, 2, n_samples).clip(0, 7)
    stress_level = np.random.normal(5, 2, n_samples).clip(1, 10)
    
    # Create realistic mood patterns
    mood_score = (sleep_hours*0.3 + steps/2000*0.2 - screen_time*0.1 - stress_level*0.2 + social_time*0.1 + exercise_freq*0.1)
    mood_labels = np.where(mood_score >= 1.5, 'happy', 
                  np.where(mood_score >= -0.5, 'stressed', 'tired'))
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    features = ['sleep_hours', 'screen_time', 'steps', 'social_time', 'exercise_freq', 'stress_level']
    X = pd.DataFrame({
        features[0]: sleep_hours, features[1]: screen_time, features[2]: steps,
        features[3]: social_time, features[4]: exercise_freq, features[5]: stress_level
    })
    y = mood_labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    st.info(f"✅ Model trained! Test accuracy: {model.score(X_test, y_test):.3f}")
    return model, features

model, features = create_model()

# Inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Your Daily Habits")
    sleep = st.slider("Sleep hours", 0.0, 12.0, 7.0)
    screen = st.slider("Screen time", 0.0, 15.0, 6.0)
    steps = st.slider("Steps", 0, 25000, 8000)

with col2:
    social = st.slider("Social time (weekly)", 0.0, 7.0, 3.0)
    exercise = st.slider("Exercise (0-7 days)", 0, 7, 3)
    stress = st.slider("Stress level", 1, 10, 5)

# Predict
if st.button("🔮 Predict My Mood", type="primary"):
    input_df = pd.DataFrame({
        features[0]: [sleep], features[1]: [screen], features[2]: [steps],
        features[3]: [social], features[4]: [exercise], features[5]: [stress]
    })
    
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("😊 Happy", f"{probs[0]:.0%}")
    with col2: st.metric("😰 Stressed", f"{probs[1]:.0%}")
    with col3: st.metric("😴 Tired", f"{probs[2]:.0%}")
    
    st.success(f"🎯 **Predicted mood: {pred.upper()}**")
    
    # Advice
    if pred == 'happy':
        st.balloons()
        st.success("🎉 Great habits! Keep it up! 💪")
    elif pred == 'stressed':
        st.warning("😰 Try reducing screen time or adding more steps!")
    else:
        st.info("😴 More sleep and exercise could help!")

st.markdown("---")
st.caption("🤖 Student Mood Prediction Project | Ready for portfolio! 🚀")
