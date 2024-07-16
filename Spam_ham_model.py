# Importing libraries and dependencies
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score ,confusion_matrix
import joblib

# Reading the data from csv file and loading to pandas dataframe
data = pd.read_csv("mail_data.csv")

# Basic exploration of the dataset
print(data.shape)
print(data.columns)
print(data.info())
print(data.head())

# Check for missing values
print("Total number of null values in both the columns")
print(data.isnull().sum())

# Check for duplicates and remove them
print("Before removing the duplicates total number ", data.duplicated().sum())
# Removing dupliactes
data.drop_duplicates(keep='first', inplace=True)
print("After removing the duplicates ", data.duplicated().sum())

print(data.head())

# Applying LabelEncoder to Spam column

# Label Encoding-> transform a categorical column into numerical form(spam=0,ham=1)
# encoder = LabelEncoder()
# data['Category']=encoder.fit_transform(data['Category'])
# OR
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

print(data.head())

# Renaming a column
data.rename(columns={'Category':'Ham'},inplace=True)
print(data.columns)
print(data.sample())

# Summary of the dataset
data.info()

# Performing Exploratory Data Analysis (EDA)

# Check distribution of 'Spam' column values
print(data['Ham'].value_counts())  
# Note: Data is imbalanced

# Check number of characters in each message
data['char_count'] = data['Message'].apply(len)  
# Check number of words
data['word_count'] = data['Message'].apply(lambda x: len(nltk.word_tokenize(x)))  
# Check number of sentences
data['sentence_count'] = data['Message'].apply(lambda x: len(nltk.sent_tokenize(x)))  

print(data.head())

# Check statistics for Ham messages
print("Exploring the characteristics of Ham messages (non-spam)")
# Display descriptive statistics for Ham messages
ham_stats = data[data['Ham'] == 0].describe()
print(ham_stats)

# Check statistics for Spam messages
print("\nExploring the characteristics of Spam messages")
# Display descriptive statistics for Spam messages
spam_stats = data[data['Ham'] == 1].describe()
print(spam_stats)


"""Data Preprocessing:
    1.Convert to lowercase
    2.Tokenize the messages(break into list of  words)
    3.Remove special symbols
    4.Remove stopwords and punctuation
    5.Apply Stemming"""

#Intialize the PorterStemmer
stemmer = PorterStemmer()

def transform(text):
    # Convert to lowercase
    text = text.lower()  

    # Tokenize the text
    text = nltk.word_tokenize(text) 

    # Remove non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  
    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  
    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(stemmer.stem(i))  

    return " ".join(y) # Returning the final text

# Add new column to store the transformed messages
data['Transformed_text'] = data['Message'].apply(transform)
print("After applying the above steps the modified dataframe")
print(data.head())

# Initialize CountVectorizer and transform the text data to numeric format (Bag of Words)
vectorizer = CountVectorizer()

# Feature extraction
x = vectorizer.fit_transform(data['Transformed_text']).toarray()
# Target encoding
y = data['Ham'].values.astype(int)

print(x.shape)
print(y.shape)

# Splitting the data into test and train data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)


# Model initializing and training

# 1.Support vector classifier(SVC)
model_svc = SVC(kernel='linear', C=1.5)
model_svc.fit(X_train, Y_train)

# 2.Logistic Regression(Binary Classification)
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)

# 3.Decision Tree Classifier
model_dtr = DecisionTreeClassifier()
model_dtr.fit(X_train, Y_train)

# Evaluating trained Models

# 1.SVC Evaluation
prediction_on_training_data = model_svc.predict(X_train)
print("SVC - Accuracy on training data: ", accuracy_score(Y_train, prediction_on_training_data))
print("SVC - Precision on training data: ", precision_score(Y_train, prediction_on_training_data))

prediction_on_test_data = model_svc.predict(X_test)
print("SVC - Accuracy on testing data: ", accuracy_score(Y_test, prediction_on_test_data))
print("SVC - Precision on testing data: ", precision_score(Y_test, prediction_on_test_data))
print("SVC - Confusion Matrix on testing data:\n", confusion_matrix(Y_test, prediction_on_test_data))

# 2.Logistic Regression evaluation
prediction_on_training_data = model_lr.predict(X_train)
print("Logistic Regression - Accuracy on training data: ", accuracy_score(Y_train, prediction_on_training_data))
print("Logistic Regression - Precision on training data: ", precision_score(Y_train, prediction_on_training_data))

prediction_on_test_data = model_lr.predict(X_test)
print("Logistic Regression - Accuracy on testing data: ", accuracy_score(Y_test, prediction_on_test_data))
print("Logistic Regression - Precision on testing data: ", precision_score(Y_test, prediction_on_test_data))
print("Logistic Regression - Confusion Matrix on testing data:\n", confusion_matrix(Y_test, prediction_on_test_data))

# 3.Decision Tree Classifier evaluation
prediction_on_training_data = model_dtr.predict(X_train)
print("Decision Tree Classifier - Accuracy on training data: ", accuracy_score(Y_train, prediction_on_training_data))
print("Decision Tree Classifier - Precision on training data: ", precision_score(Y_train, prediction_on_training_data))

prediction_on_test_data = model_dtr.predict(X_test)
print("Decision Tree Classifier - Accuracy on testing data: ", accuracy_score(Y_test, prediction_on_test_data))
print("Decision Tree Classifier - Precision on testing data: ", precision_score(Y_test, prediction_on_test_data))
print("Decision Tree Classifier - Confusion Matrix on testing data:\n", confusion_matrix(Y_test, prediction_on_test_data))

# Saving Trained model and Vectorizer to files
joblib.dump(model_svc, 'SVC.pkl')  # saving the trained model
joblib.dump(vectorizer, 'countVectorizer.pkl')  # saving the fitted vectorizer
print("Model saved as SVC.pkl and countVectorizer.pkl")
