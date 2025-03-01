import pandas as pd
import seaborn as ssn
import matplotlib.pylab as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# Reading data from Database
data = pd.read_csv("./data_english.csv")
# splitting data into test and train
x_train, x_test, y_train, y_test = model_selection.train_test_split(data["Description"], data["Category"], test_size=0.2, random_state=6)


# Encoding Categories
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

label_to_number_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Description vectorization TF_IDF
tfidf_vector = TfidfVectorizer()
tfidf_vector.fit(data["Description"])
x_train_tfidf = tfidf_vector.transform(x_train)
x_test_tfidf = tfidf_vector.transform(x_test)

# Train the SVM model
svm_model = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")
svm_model.fit(x_train_tfidf, y_train)

# Prediction phase
svm_predictions = svm_model.predict(x_test_tfidf)

# Accuracy and evaluation
print("SVM Accuracy Score -> ", accuracy_score(svm_predictions, y_test) * 100)
print("SVM Precision Score -> ", precision_score(y_test, svm_predictions, average='weighted') * 100)
print("Recall Score -> ", recall_score(y_test, svm_predictions, average='weighted') * 100)
print("F1 Score -> ", f1_score(y_test, svm_predictions, average='weighted') * 100)


svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
plt.figure(figsize=(15, 15))
ssn.heatmap(svm_confusion_matrix, annot=True, cmap='Greens', fmt='d', cbar=False, yticklabels=label_to_number_mapping
            , xticklabels=label_to_number_mapping)
plt.title("SVM Model")
plt.xlabel("Predicted Categories")
plt.title("Actual Categories")
plt.tight_layout()
plt.show()