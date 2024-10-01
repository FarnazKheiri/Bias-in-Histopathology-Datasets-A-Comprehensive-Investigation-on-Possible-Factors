from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
from KimiaNet_Featute_Filtering import KimiaNet_Features_Filtering

######################################## load filtered test data
x_test, y_test_center, y_test_cancer = KimiaNet_Features_Filtering("test")


######################################## search the test samples into the test set using KNN concept
def test_in_test(k , features , label):
    classifier = KNeighborsClassifier(n_neighbors=k)
    predicted_labels = [None] * len(features)

    for i in range(len(features)):
        print(i)

        ## define arrays
        x_temple = list(features).copy()
        label_temple = list(label).copy()

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
    return predicted_labels




################################################ cancer classification
cancer_predicted_labels = test_in_test( k =3, features = x_test, label = y_test_cancer)
metrics.accuracy_score(y_test_cancer,cancer_predicted_labels)

################################################ center classification
center_predicted_labels = test_in_test( k =3, features = x_test, label = y_test_cancer)
metrics.accuracy_score(y_test_center,center_predicted_labels)


