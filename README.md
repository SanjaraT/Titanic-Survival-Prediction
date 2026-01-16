📌 Overview

This project applies machine learning techniques to predict passenger survival on the Titanic dataset. The workflow follows a complete ML pipeline including preprocessing, feature scaling, model training, hyperparameter tuning, and evaluation.

📂 Dataset

Source: Kaggle Titanic Competition
Files used:
train.csv – training data with labels
test.csv – unseen data for final prediction

🔧 Preprocessing

Handled missing values
Encoded categorical features
Checked class balance
Applied StandardScaler for models sensitive to feature scale
Split training data into train, validation, and test sets

🤖 Models Used

Logistic Regression (with hyperparameter tuning)
Random Forest
Gradient Boosting (final model)

📊 Evaluation

Models were evaluated using validation accuracy and classification metrics
Final model achieved ~81% validation accuracy
Test data was used only for prediction, as labels are unavailable