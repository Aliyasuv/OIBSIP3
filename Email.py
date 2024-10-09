import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
url = "https://storage.googleapis.com/kagglesdsdata/datasets/483/982/spam.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241009%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241009T104303Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9ec509cd64a8cc1555ff18f62aa0c01610b6ecb5a9b95a8dc01f19b5ada751979570f9a99d5d55595ad2f992275475f3bb1b94b59c7d787ea98f38db513566bb6d4263405d31c1b79a0de6507433dad69b237ff9773a046b4736ff1470cd225583e1cacd428890621009ab4186ae5e8adf450c462ad92d9cf3da5f16c58b7f08739c8e5e7ae5b8f3e9d5a1b80b7956e324e174df3df802ef8822a9fa3c92e2d250a63244062240d47210f2a647568a89fc625c8eea2ac0234727fcedaf13bcada138b218fcba71446c9be90356a5f09cde4c7ab6514a5642902e5881cea6fabb2270395dadc798bef022ec8ecadaf2226162267c7a1b5a7f41ad15d4cad57af2"
spam_data = pd.read_csv(url, encoding='latin-1')

# Display the first few rows
print(spam_data.head())

# Drop unnecessary columns
spam_data = spam_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
spam_data = spam_data.rename(columns={"v1": "label", "v2": "text"})

# Map labels to binary values
spam_data['label'] = spam_data['label'].map({'ham': 0, 'spam': 1})

# Data visualization
sns.countplot(x='label', data=spam_data)
plt.title('Spam vs. Ham')
plt.show()

# Prepare the data
X = spam_data['text']
y = spam_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
