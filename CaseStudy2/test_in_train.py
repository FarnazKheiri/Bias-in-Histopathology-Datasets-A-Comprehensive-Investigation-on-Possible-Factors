
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def test_in_train(k ,test_features, test_labels, train_features, train_labels):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_features, train_labels)
    pred_labels = classifier.predict(test_features)
    acc = metrics.accuracy_score(test_labels,pred_labels)
    return acc