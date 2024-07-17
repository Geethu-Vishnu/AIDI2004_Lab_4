import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # or any other model
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv(r'/Users/geethuvishnu/Downloads/AIDI2004_lAB4/data/Fish.csv')

# Data Preprocessing
X = data.drop('Species', axis=1)
y = data['Species']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(model, '/Users/geethuvishnu/Downloads/AIDI2004_lAB4/app/model.pkl')