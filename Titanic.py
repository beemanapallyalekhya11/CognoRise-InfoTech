import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = "C:/Users/alekh/Desktop/cognorise task/tested.csv"  # Update to the correct file path
df = pd.read_csv(data)

# Display the first few rows of the dataset
print(df.head())

# Display column names
print("Column Names:")
print(df.columns)

# Total number of people who traveled on the Titanic
total_people = df.shape[0]
print(f"Total number of people who traveled on the Titanic: {total_people}")

# Column selection
ages_and_sex = df[['Age', 'Sex']]
print("Selected Columns (Age, Sex):")
print(ages_and_sex.head())

# Selection of a single column
ages = df['Age']
print("Age Column:")
print(ages.head())

# Filtering rows by gender
df_female = df[df['Sex'] == 'female']
print("Female Rows:")
print(df_female.head())

# Data aggregation
average_age = df['Age'].mean()
print(f"Average Age: {average_age}")

# Filtering rows where the passenger survived
df_survived = df[df['Survived'] == 1]
print("Survived Rows:")
print(df_survived.head())
print(f"Count of Survived Rows: {df_survived.count()}")
print(f"Number of Survived Rows: {df_survived.shape[0]}")
print(f"Average Age of Survived: {df_survived['Age'].mean()}")

print("DataFrame info:")
print(df.info())

# Joining two DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Ana', 'Bruno', 'Carla']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'surname': ['Silva', 'Souza', 'Santos']})
df_merged = pd.merge(df1, df2, on='ID', how='inner')
print("Merged DataFrame:")
print(df_merged)

# SQL-like comments for joins
'''
SQL

-- Inner Join
SELECT * FROM df1
INNER JOIN df2 ON df1.ID = df2.ID

-- Left Join
SELECT * FROM df1
LEFT JOIN df2 ON df1.ID = df2.ID

-- Right Join
SELECT * FROM df1
RIGHT JOIN df2 ON df1.ID = df2.ID

-- Full Join
SELECT * FROM df1
FULL JOIN df2 ON df1.ID = df2.ID
'''

# Data Preprocessing
# Handling missing values (example: filling missing Age values with the mean)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop columns that won't be used
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Titanic survival prediction:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Titanic Survival Prediction\nTotal Passengers: {total_people}')
plt.show()

# Pie chart
survival_counts = df['Survived'].value_counts()
labels = 'Not Survived', 'Survived'
colors = ['red', 'blue']
explode = (0, 0.1)

plt.figure(figsize=(8, 8))
plt.pie(survival_counts, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Survivors (1) and the dead (0)')
plt.axis('equal')
plt.show()

# Bar chart
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df, palette=['red', 'blue'])
plt.title('Survivors (1) and the dead (0)')
plt.xlabel('Survived')
plt.ylabel('Quantity')
plt.show()
