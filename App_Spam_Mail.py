import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
import random
st.set_page_config(page_title = 'AI Spam Detector', layout = 'wide')
st.title("📧 AI Spam Mail Detection System")
raw_mail_data = pd.read_csv("mail_data.csv")
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),"")
mail_data.loc[mail_data['Category'] == 'spam', "Category",] = 0
mail_data.loc[mail_data['Category'] == "ham", 'Category',] = 1
X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
model = LogisticRegression()
model.fit(X_train_features, Y_train)
st.sidebar.header("📊 Spam Analytics")
total_emails = len(mail_data)
spam_count = sum(mail_data['Category'] == 0)
ham_count = sum(mail_data['Category'] == 1)
st.sidebar.metric("Total emails: ", total_emails)
st.sidebar.metric("Spam emails: ", spam_count)
st.sidebar.metric("Safe emails: ", ham_count)
spam_percent = round((spam_count/total_emails) * 100, 2)
st.sidebar.write("Spam percentage: ", spam_percent, " %")
st.subheader("📊 Spam Analytics Dashboard")
chart_data = pd.DataFrame({"Type" : ["Spam", "Safe"], "Count" : [spam_count, ham_count]})
fig = px.pie(chart_data, values = "Count", names = "Type", title = "Spam vs Safe Emails")
st.plotly_chart(fig, use_container_width = True)
st.subheader("📥 Email Inbox Simulator")
sample_emails = mail_data.sample(8)
for i, row in sample_emails.iterrows():
  col1, col2 = st.columns([1, 8])
  with col1:
    if row['Category'] == 0:
      st.markdown("🚨")
    else:
      st.markdown("📩")
  with col2:
    with st.expander("Open Email"):
      st.write(row['Message'])
st.subheader("🔍 Live Email Security Scanner")
col1, col2 = st.columns([3, 1])
with col1:
    input_mail = st.text_area("📩 Paste incoming email content", height = 150)
with col2:
    scan_button = st.button("🔎 Scan Email")
st.markdown("-----")
if scan_button:
  if input_mail.strip() == '':
    st.warning("⚠️ Please enter email content first.")
  else:
    with st.spinner("Scanning email for spam threats"):
         input_features = vectorizer.transform([input_mail])
         prediction = model.predict(input_features)
         probability = model.predict_proba(input_features)
         spam_prob = probability[0][0]
         ham_prob = probability[0][1]
    st.subheader("🧠AI Security Verdict")
    if prediction[0] == 0:
       st.error("🚨 Spam Email Detected")
    else:
       st.success("✅ Safe Email")
    st.subheader("📊 Spam Probability")
    st.progress(float(spam_prob))
    st.write("Spam probability: ", round(spam_prob * 100, 2), " %")
    st.write("Safe probability: ", round(ham_prob * 100, 2), " %")
    st.subheader("🧠 Important Words")
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    email_words = input_mail.split()
    word_scores = []
    for word in email_words:
        if word in feature_names:
           idx = list(feature_names).index(word)
           word_scores.append((word, weights[idx]))
    word_scores = sorted(word_scores, key = lambda x: abs(x[1]), reverse = True)[:10]
    explain_df = pd.DataFrame(word_scores, columns = ["Word", "Importance"])
    if not explain_df.empty:
      st.bar_chart(explain_df.set_index("Word"))
    else:
      st.write("No important words detected")
    st.subheader("🎯 AI Confidence Meter")
    confidence = spam_prob * 100
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=confidence,
    title={'text': "Spam Threat Level"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'thickness': 0.3},
        'steps': [
            {'range': [0, 30], 'color': "green"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"}
        ]
    }
))

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔥 Spam Word Heatmap")

    if not explain_df.empty:
       heatmap_fig = px.imshow(
        [explain_df["Importance"].values],
        labels=dict(x="Words", y="Impact", color="Spam Score"),
        x=explain_df["Word"].values
    )
       st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
       st.write("No heatmap data available")
if scan_button:
  st.subheader("📡 Live Spam Threat Radar")

  radar_data = pd.DataFrame({
        "Angle": list(range(0, 360, 10)),
        "Threat": [random.randint(1,100) for i in range(36)]
    })

  radar_fig = px.line_polar(
        radar_data,
        r="Threat",
        theta="Angle",
        line_close=True
    )

  st.plotly_chart(radar_fig, use_container_width=True)
