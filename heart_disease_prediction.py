import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart_disease_uci.csv')

df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df = df.dropna()

X = df[['age','trestbps','chol','thalch','oldpeak']]
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Heart Disease Accuracy: {accuracy * 100:.2f}%')