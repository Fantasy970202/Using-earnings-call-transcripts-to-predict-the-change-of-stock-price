
from gensim.models import Word2Vec, word2vec
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

def FunctionText2Vec(inpTextData):
    # Converting the text to numeric data
    X = vectorizer.transform(inpTextData)
    CountVecData = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    # Creating empty dataframe to hold sentences
    W2Vec_Data = pd.DataFrame()
    # Looping through each row for the data
    for i in range(CountVecData.shape[0]):
        # initiating a sentence with all zeros
        Sentence = np.zeros(300)
        # Looping thru each word in the sentence and if its present in
        # the Word2Vec model then storing its vector
        for word in WordsVocab[CountVecData.iloc[i, :]]:
            # print(word)
            if word in GoogleModel.key_to_index.keys():
                Sentence = Sentence + GoogleModel[word]
        # Appending the sentence to the dataframe
        W2Vec_Data = W2Vec_Data.append(pd.DataFrame([Sentence]))
    return (W2Vec_Data)
# Naive Bayes
def Naive_Bayes(X_train,y_train,X_test, y_test, X, y):
    print("Naive_Bayes:")
# GaussianNB is used in Binomial Classification
# MultinomialNB is used in multi-class classification
    clf = GaussianNB()
    #clf = MultinomialNB()
# Printing all the parameters of Naive Bayes
    NB=clf.fit(X_train,y_train)
    prediction=NB.predict(X_test)
    #print(prediction)

# Measuring accuracy on Testing Data
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
    F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))
# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
    Accuracy_Values=cross_val_score(NB, X , y, cv=10, scoring='f1_weighted')
    print('\nAccuracy values for 5-fold Cross Validation:\n',Accuracy_Values)
    print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))
    print("Naive_Bayes model complete.\n")


def Random_Forest(X_train, y_train, X_test, y_test):
    print("Random Forest:")
    clf = RandomForestClassifier(n_estimators = 100)
    # Creating the model on Training Data
    RF = clf.fit(X_train, y_train)
    prediction = RF.predict(X_test)

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

    # Printing the Overall Accuracy of the model
    F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
    print("Random Forest model complete.\n")

def XGBoost(X_train, y_train, X_test, y_test):
    print("XGBoost:")
    clf = XGBClassifier(learning_rate=0.01, n_estimators=100, n_jobs=-1)
    # Creating the model on Training Data
    RF = clf.fit(X_train, y_train)
    prediction = RF.predict(X_test)

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

    # Printing the Overall Accuracy of the model
    F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
    print("XFBoost model complete.\n")

def SVM(X_train, y_train, X_test, y_test):
    print("SVM:")
    clf = SVC()
    # Creating the model on Training Data
    RF = clf.fit(X_train, y_train)
    prediction = RF.predict(X_test)

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

    # Printing the Overall Accuracy of the model
    F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
    print("SVM model complete.\n")

def KNN(X_train, y_train, X_test, y_test):
    print("KNN:")
    clf = KNeighborsClassifier(n_neighbors=15)
    # Creating the model on Training Data
    KNN = clf.fit(X_train, y_train)
    prediction = KNN.predict(X_test)

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

    # Printing the Overall Accuracy of the model
    F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
    print("KNN model complete.\n")

    # Importing cross validation function from sklearn
    from sklearn.model_selection import cross_val_score

    # Running 10-Fold Cross validation on a given algorithm
    # Passing full data X and y because the K-fold will split the data and automatically choose train/test
    # Accuracy_Values=cross_val_score(KNN, X , y, cv=10, scoring='f1_weighted')
    # print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
    # print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

    # Plotting the feature importance for Top 10 most important columns
    # There is no built-in method to get feature importance in KNN

def Logistic_Regression(X_train, y_train, X_test, y_test):
    print("Logistic_Regression:")
    # choose parameter Penalty='l1' or C=1
    # choose different values for solver 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    clf = LogisticRegression(C=10, penalty='l2', solver='newton-cg')

    # Printing all the parameters of logistic regression
    # print(clf)

    # Creating the model on Training Data
    LOG = clf.fit(X_train, y_train)

    # Generating predictions on testing data
    prediction = LOG.predict(X_test)
    # Printing sample values of prediction in Testing data
    TestingData = pd.DataFrame(data=X_test, columns=Predictors)
    #print(TestingData.head())

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(prediction, y_test))

    ## Printing the Overall Accuracy of the model
    F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
    print("Logistic_Regression model complete.\n")


    ## Importing cross validation function from sklearn
    # from sklearn.model_selection import cross_val_score

    ## Running 10-Fold Cross validation on a given algorithm
    ## Passing full data X and y because the K-fold will split the data and automatically choose train/test
    # Accuracy_Values=cross_val_score(LOG, X , y, cv=10, scoring='f1_weighted')
    # print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
    # print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

def Adaboost(X_train, y_train, X_test, y_test):
    print("Adaboost:")
    # Choosing Decision Tree with 1 level as the weak learner
    DTC = DecisionTreeClassifier(max_depth=2)
    clf = AdaBoostClassifier(n_estimators=20, base_estimator=DTC, learning_rate=0.01)
    # Creating the model on Training Data
    AB = clf.fit(X_train, y_train)
    prediction = AB.predict(X_test)

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

    # Printing the Overall Accuracy of the model
    F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
    print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
    print("Adaboost model complete.\n")


def FunctionPredictUrgency(inpText, X, y):
    print("Predicting results of new texts:")
    # Generating the Glove word vector embeddings
    clf = KNeighborsClassifier(n_neighbors=15)
    FinalModel = clf.fit(X, y)
    X = FunctionText2Vec(inpText)
    # print(X)

    # If standardization/normalization was done on training
    # then the above X must also be converted to same platform
    # Generating the normalized values of X
    X = PredictorScalerFit.transform(X)


    # Generating the prediction using Naive Bayes model and returning
    Prediction = FinalModel.predict(X)
    Result = pd.DataFrame(data=inpText, columns=['Text'])
    Result['Prediction'] = Prediction
    return (Result)
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Reading the data
    Data = pd.read_csv('complete_union.csv')
    # Ticket Data
    corpus = Data['cleaned_transcript'].values
    # Creating the vectorizer
    vectorizer = CountVectorizer()
    # Converting the text to numeric data
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())
    # Preparing Data frame For machine learning
    # Priority column acts as a target variable and other columns as predictors
    CountVectorizedData = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    CountVectorizedData['label_day_one'] = Data['Day1']
    print("Data shape:")
    print(CountVectorizedData.shape)
    # Loading the word vectors from Google trained word2Vec model
    GoogleModel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, )
    # Defining a function which takes text input and returns one vector for each sentence


    # 列名
    WordsVocab = CountVectorizedData.columns[:-1]
    W2Vec_Data = FunctionText2Vec(Data['cleaned_transcript'])

    # Checking the new representation for sentences
    # print(W2Vec_Data.shape)
    # print(Data['Day1'])
    W2Vec_Data.reset_index(inplace=True, drop=True)
    x = CountVectorizedData['label_day_one']
    W2Vec_Data['label_day_one'] = x
    # Assigning to DataForML variable
    DataForML = W2Vec_Data
    print("GoogleModel data shape:")
    print(DataForML.shape)
    # print(DataForML.head(10))

    # Separate Target Variable and Predictor Variables
    TargetVariable = DataForML.columns[-1]
    Predictors = DataForML.columns[:-1]

    X = DataForML[Predictors].values
    y = DataForML[TargetVariable].values

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)
    PredictorScaler = MinMaxScaler()

    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(X)

    # Generating the standardized values of X
    X = PredictorScalerFit.transform(X)

    # Split the data into training and testing set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)

    Naive_Bayes(X_train,y_train,X_test, y_test, X, y)
    KNN(X_train, y_train, X_test, y_test)
    Logistic_Regression(X_train, y_train, X_test, y_test)
    Adaboost(X_train, y_train, X_test, y_test)
    Random_Forest(X_train, y_train, X_test, y_test)
    XGBoost(X_train, y_train, X_test, y_test)
    SVM(X_train, y_train, X_test, y_test)
    test = ["help to review the issue", "Please help to resolve system issue"]
    print(FunctionPredictUrgency(test, X, y))