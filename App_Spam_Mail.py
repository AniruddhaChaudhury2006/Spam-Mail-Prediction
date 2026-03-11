import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st
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
st.subheader("📥 Email Inbox Simulator")
sample_emails = mail_data.sample(5)
for msg in sample_emails['Message']:
  with st.expander("Open Email"):
    st.write(msg)
st.subheader("🔍 Check an Email")
input_mail = st.text_area("Paste email content:")
if st.button("Detect Spam"):
  input_features = vectorizer.transform([input_mail])
  prediction = model.predict(input_features)
  probability = model.predict_proba(input_features)
  spam_prob = probability[0][0]
  ham_prob = probability[0][1]
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
  st.bar_chart(explain_df.set_index("Word"))
