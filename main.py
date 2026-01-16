import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv("titanic_train.csv")
df_test = pd.read_csv("titanic_test.csv")

# print(df.head())
# print(df.shape)
# print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(),inplace=True)
df.drop(columns=["PassengerId","Name","Cabin","Ticket",],inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'],drop_first=True)


df_test['Age'].fillna(df_test['Age'].median(),inplace=True)
df_test.drop(columns=["PassengerId","Name","Cabin","Ticket",],inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0],inplace=True)
df_test['Sex'] = df_test['Sex'].map({'male':0, 'female':1})
df_test = pd.get_dummies(df_test, columns=['Embarked'],drop_first=True)

# print(df.head())
# print(df_test.head())

X = df.drop('Survived', axis = 1)
y = df['Survived']

X, df_test = X.align(df_test, axis=1, fill_value=0)

#Split
X_train, X_val, y_train, y_val = train_test_split(
    X,y, test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(df_test)

#Correlation
# scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# corr_matrix = scaled_df.corr()
# plt.figure(figsize=(12,10))
# sns.heatmap(
#     corr_matrix, annot=True,
#     fmt = ".2f", cmap="coolwarm"
# )
# plt.title("Correlation Matrix")
# plt.show()

# Class Distribution
# plt.figure(figsize=(5,4))
# df["Survived"].value_counts().plot(kind = "bar",)
# plt.xticks([0,1],["Yes","No"],rotation = 0)
# plt.ylabel("Count")
# plt.title("Distribution")
# plt.show()

#LOGISTIC REGRESSION
# lr = LogisticRegression(
#     max_iter=1000,
#     random_state=42
# )
# param_grid = {
#     'C':[0.01,0.1,1,10],
#     'penalty':['l2'],
#     'solver':['liblinear','lbfgs']
# }
# grid_search = GridSearchCV(
#     estimator=lr,
#     param_grid=param_grid,
#     cv = 5,
#     scoring = 'f1',
#     n_jobs = -1
# )

# grid_search.fit(X_train_scaled,y_train)
# best_lr = grid_search.best_estimator_

#prediction
# y_val_pred = best_lr.predict(X_val_scaled)
# val_acc = accuracy_score(y_val, y_val_pred)
# print(classification_report(y_val,y_val_pred))

#confusison_matrix
# cm_lr = confusion_matrix(y_val,y_val_pred)
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_lr,annot=True, fmt ='d',cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

#RANDOM FOREST
# rf = RandomForestClassifier(
#     random_state=42,
#     n_jobs=-1
# )
# param_grid={
#     'n_estimators':[100,200,300],
#     'max_depth':[None,5,10],
#     'min_samples_split':[2,5],
#     'min_samples_leaf':[1,2]
# }
# grid_rf = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv = 5,
#     scoring = 'f1',
#     n_jobs = -1
# )
# grid_rf.fit(X_train_scaled,y_train)
# best_rf = grid_rf.best_estimator_

# #prediction
# y_val_pred = best_rf.predict(X_val_scaled)
# val_acc = accuracy_score(y_val, y_val_pred)
# print(classification_report(y_val,y_val_pred))

# #confusison_matrix
# cm_rf = confusion_matrix(y_val,y_val_pred)
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_rf,annot=True, fmt ='d',cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

#GRADIENT BOOSTING
gb = GradientBoostingClassifier(
    random_state=42
)
param_grid ={
    'n_estimators':[100,200],
    'learning_rate':[0.05,0.1],
    'max_depth':[3,5],
    'subsample':[0.8,1.0]
}
grid_gb = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv = 5,
    scoring = 'f1',
    n_jobs = -1
)
grid_gb.fit(X_train_scaled,y_train)
best_gb = grid_gb.best_estimator_

#prediction
y_val_pred_gb = best_gb.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_val_pred_gb)
print(classification_report(y_val,y_val_pred_gb))

#confusison_matrix
cm_gb = confusion_matrix(y_val,y_val_pred_gb)
plt.figure(figsize=(5,4))
sns.heatmap(cm_gb,annot=True, fmt ='d',cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

final_model = GradientBoostingClassifier(
    **grid_gb.best_params_,
    random_state=42
)

final_model.fit(X, y)