from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics



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







