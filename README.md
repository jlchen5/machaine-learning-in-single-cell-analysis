# machine-learning-in-single-cell-analysis



In this example code, we first load the scRNA-seq data from a CSV file and separate the labels from the features. We then split the data into training and testing sets and standardize the features using scikit-learn's StandardScaler class. Next, we perform principal component analysis (PCA) for dimensionality reduction using scikit-learn's PCA class. We then train a support vector machine (SVM) model using scikit-learn's SVC class and predict the labels for the test set. Finally, we calculate the accuracy score using scikit-learn's accuracy_score function.
