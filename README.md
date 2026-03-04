# Spam Mail Classification

## Project Overview
This project implements a machine learning model to classify emails as either 'spam' or 'ham' (not spam). The goal is to build a predictive model that can effectively distinguish between legitimate emails and unsolicited junk mail.

## Technologies Used
*   **Python**: Programming language.
*   **Pandas**: For data manipulation and analysis.
*   **Numpy**: For numerical operations.
*   **Scikit-learn**: For machine learning tasks, including:
    *   `train_test_split`: To divide data into training and testing sets.
    *   `TfidfVectorizer`: To convert text data into numerical feature vectors.
    *   `LogisticRegression`: The classification model used.
    *   `accuracy_score`: To evaluate model performance.

## Dataset
The dataset used for this project is `mail_data.csv`, which contains two columns: `Category` (indicating 'spam' or 'ham') and `Message` (the content of the email).

## Setup and Installation
To run this project, you need to have Python and the following libraries installed:

```bash
pip install pandas numpy scikit-learn
```

## Data Preprocessing
1.  **Loading Data**: The `mail_data.csv` file is loaded into a Pandas DataFrame.
2.  **Handling Missing Values**: Empty strings are replaced with null values, which are then handled (in this case, by replacing them with empty strings, although the dataset appears clean).
3.  **Label Encoding**: The 'Category' column is converted from categorical ('spam', 'ham') to numerical (0 for spam, 1 for ham).
4.  **Splitting Data**: The dataset is split into training and testing sets (80% training, 20% testing).
5.  **Feature Extraction**: `TfidfVectorizer` is used to transform the text `Message` data into numerical feature vectors. This process converts text into a matrix of TF-IDF features.

## Model Training
A Logistic Regression model is trained on the TF-IDF transformed training data. Logistic Regression is a suitable choice for binary classification problems like spam detection due to its efficiency and interpretability.

## Model Evaluation
The model's performance is evaluated using accuracy scores on both the training and test datasets. The accuracy indicates how well the model predicts the correct category of emails.

*   **Accuracy on Training Data**: 0.9677 (approximately 96.77%)
*   **Accuracy on Test Data**: 0.9668 (approximately 96.68%)

## Usage
To predict whether a new email is spam or ham, follow these steps:

1.  Provide the email content as input.
2.  Transform the input email using the same `TfidfVectorizer` fitted on the training data.
3.  Use the trained `LogisticRegression` model to make a prediction.

### Example Prediction
```python
input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's'"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
  print("Ham mail")
else:
  print("Spam mail")
```

## Results
The model achieved high accuracy on both training and test data, demonstrating its effectiveness in classifying spam and ham emails.
