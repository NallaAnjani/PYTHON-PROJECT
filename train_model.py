import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df.dropna(inplace=True)

# Encode labels (spam = 1, ham = 0)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Vectorize text messages
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest (Bagging)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, 'rf_model.pkl')

# Train AdaBoost (Boosting)
ab = AdaBoostClassifier()
ab.fit(X_train, y_train)
joblib.dump(ab, 'ab_model.pkl')

# Save the text vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Models trained and saved successfully!")
