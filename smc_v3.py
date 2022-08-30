import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle

df = pd.read_csv('spam.csv')
#print(df.head())
#print(df.shape)

X = df['Message']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

with open('email_classification.pickle', 'wb') as f:
    pickle.dump(model, f)


