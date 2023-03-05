# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load scRNA-seq data
data = pd.read_csv("scRNAseq_data.csv")

# separate labels from features
X = data.drop(["label"], axis=1)
y = data["label"]

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# perform PCA for dimensionality reduction
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# train SVM model
svm = SVC(kernel="linear", C=0.1)
svm.fit(X_train, y_train)

# predict labels for test set
y_pred = svm.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
