# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# titanic = sns.load_dataset('titanic')
# print(titanic.head())


# Load the dataset
titanic_df = pd.read_csv('titanic.csv')

# Data cleaning and preprocessing
titanic_df.dropna(inplace=True)
titanic_df['Age'] = pd.cut(titanic_df['Age'], bins=[0, 18, 35, 60, np.inf], labels=['Child', 'Adult', 'Middle-Aged', 'Senior'])

# Bar plot: Survival rate by sex
sns.set()
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()

# Pie chart: Class distribution
plt.figure(figsize=(8, 6))
titanic_df['Pclass'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()

# Dot plot: Age vs. Survival
plt.figure(figsize=(10, 8))
sns.stripplot(x='Age', y='Survived', data=titanic_df, jitter=True)
plt.title('Age vs. Survival')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.show()

# Bar plot: Survival rate by class
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic_df)
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.show()