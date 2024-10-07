from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut, cross_val_predict



######################################## search the test samples into the test set using KNN concept
def test_in_test(k , features , labels):
    classifier = KNeighborsClassifier(n_neighbors=k)
    predicted_labels = [None] * len(features)

    for i in range(len(features)):
        print(i)

        ## define arrays
        x_temple = list(features).copy()
        label_temple = list(labels).copy()

        ## excluding
        x_temple.pop(i)
        label_temple.pop(i)
        ###
        tst = np.array(features)[i]
        tst = tst.reshape(1, -1)
        ## classifier
        classifier.fit(x_temple, label_temple)
        predicted_label = classifier.predict(tst)

        predicted_labels[i] = predicted_label

        acc = metrics.accuracy_score(labels,predicted_labels)
    return acc


def test_in_test(k , features , labels):
    classifier = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')


    # Fit the classifier once on the entire dataset
    classifier.fit(features, labels)


    # Initialize LeaveOneOut cross-validation
    loo = LeaveOneOut()

    predicted_labels = np.empty(len(features), dtype=object)

    # LeaveOneOut iteration
    for train_index, test_index in loo.split(features):
        print(test_index[0])

        # Get train and test data according to LeaveOneOut

        x_train, x_test_loo = np.array(features)[train_index], np.array(features)[test_index]
        y_train = np.array(labels)[train_index]

        # Predict the test sample
        label_pred = classifier.predict(x_test_loo)


        # Store the prediction
        predicted_labels[test_index[0]] = label_pred
        acc = metrics.accuracy_score(labels,predicted_labels)
    return acc







