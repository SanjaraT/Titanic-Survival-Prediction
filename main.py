import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_train.csv")
# print(df.head())
# print(df.shape)

df['Age'].fillna(df['Age'].median(),inplace=True)
df.drop(columns=["PassengerId","Name","Cabin","Ticket",],inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'],drop_first=True)
print(df.head())

#Correlation
corr_matrix = df.corr()

plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix, annot=True,
    fmt = ".2f", cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()


# print(df.isnull().sum())



