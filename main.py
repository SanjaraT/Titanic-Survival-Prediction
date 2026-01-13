import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic_train.csv")
df_test = pd.read_csv("titanic_test.csv")
# print(df.head())
# print(df.shape)

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

print(df.head())
print(df_test.head())

#Correlation
# corr_matrix = df.corr()
# plt.figure(figsize=(12,10))
# sns.heatmap(
#     corr_matrix, annot=True,
#     fmt = ".2f", cmap="coolwarm"
# )
# plt.title("Correlation Matrix")
# plt.show()

#Class Distribution
# plt.figure(figsize=(5,4))
# df["Survived"].value_counts().plot(kind = "bar",)
# plt.xticks([0,1],["Yes","No"],rotation = 0)
# plt.ylabel("Count")
# plt.title("Distribution")
# plt.show()




# print(df.isnull().sum())



