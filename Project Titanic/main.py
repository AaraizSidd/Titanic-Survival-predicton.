#importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data from csv file to Panda dataframe
titanic_data = pd.read_csv('titanic_data.csv')

# Print the first 5 rows of the dataset
print(titanic_data.head())

print(titanic_data.shape)

# show info() without printing its return value
titanic_data.info()

print(titanic_data.isnull().sum())

#handling missing data
titanic_data.drop(columns='Cabin', errors='ignore', inplace=True)

# avoid inplace on a Series — assign the filled Series back to the DataFrame
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
# filepath: c:\Users\aaaraaaiiz\Desktop\Project Titanic\main.py
#importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data from csv file to Panda dataframe
titanic_data = pd.read_csv('titanic_data.csv')

# Print the first 5 rows of the dataset
print(titanic_data.head())

print(titanic_data.shape)

# show info() without printing its return value
titanic_data.info()

print(titanic_data.isnull().sum())

#handling missing data
titanic_data.drop(columns='Cabin', errors='ignore', inplace=True)

# avoid inplace on a Series — assign the filled Series back to the DataFrame
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())


print(titanic_data['Embarked'].mode([0]))


titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])  

print(titanic_data.isnull().sum())

#data analysis and visualization
print(titanic_data.describe())

#survival count
print(titanic_data['Survived'].value_counts())

# data visualization
sns.set()

# make a count plot for 'Survived' column — use keyword for the x argument and don't print the Axes
sns.countplot(x='Survived', data=titanic_data)
plt.show()

sns.countplot(x='Sex', data=titanic_data)
plt.show()

sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.show()

sns.countplot(x='Pclass', data=titanic_data)
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.show()

#data preprocessing
titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()

# converting categorical columns into numerical columns
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print(titanic_data.head())

#separating data and labels
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']
print(X)
print(Y)

#splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#model training
model = LogisticRegression()
model.fit(X_train, Y_train) 

#model evaluation
#accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data : ', training_data_accuracy)
#accuracy on test data
X_test_prediction = model.predict(X_test)       
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data : ', test_data_accuracy)



