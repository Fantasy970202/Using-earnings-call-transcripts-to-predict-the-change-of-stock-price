from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

data = pd.read_csv('complete_union.csv')
X_train,X_test,Y_train, Y_test = train_test_split(data["cleaned_transcript"], data["Day1"], test_size=0.20, random_state=30)
print("Train:",X_train.shape,Y_train.shape,"Test:",(X_test.shape,Y_test.shape))
print("TFIDF Vectorizer……")
try:
    vectorizer= TfidfVectorizer()
    tf_x_train = vectorizer.fit_transform(X_train)
    tf_x_test = vectorizer.transform(X_test)
    print(tf_x_train)
    print("TFIDF vectorizing complete.\n")
except:
    print("TFIDF Vectorization error\n")

print("SVM modeling...")
try:
    clf = LinearSVC(random_state=0)
    clf.fit(tf_x_train,Y_train)
    y_test_pred=clf.predict(tf_x_test)
    report=classification_report(Y_test, y_test_pred)
    print(report)
    print('\n')
except:
    print("SVM modeling error")


print("LogisticRegression modeling..")
try:
    clf = LogisticRegression(max_iter=1000,solver="saga")
    clf.fit(tf_x_train,Y_train)
    y_test_pred=clf.predict(tf_x_test)
    report=classification_report(Y_test, y_test_pred)
    print(report)
    print('\n')
except:
    print("LogisticRegression modeling error")

print("RandomForest  modeling..")
try:
    rf = RandomForestClassifier()
    rf.fit(tf_x_train, Y_train)
    y_test_pred = rf.predict(tf_x_test)
    report = classification_report(Y_test, y_test_pred)
    print(report)
except:
    print("RandomForest modeling error")

print("XGBoost  modeling..")
try:
    rf = XGBClassifier()
    rf.fit(tf_x_train, Y_train)
    y_test_pred = rf.predict(tf_x_test)
    report = classification_report(Y_test, y_test_pred)
    print(report)
except:
    print("XGBoost modeling error")