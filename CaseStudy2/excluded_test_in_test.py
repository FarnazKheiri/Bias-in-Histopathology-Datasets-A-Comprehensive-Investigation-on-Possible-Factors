from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics


def excluded_test_in_train(k,test_features,test_labels,test_slide_names):

    classifier = KNeighborsClassifier(n_neighbors=k)


    predicted_test_labels = [None] * len(test_features)
    unique_test_slide_names = np.unique(test_slide_names)

    # Loop through unique slide names
    for i, slide in enumerate(unique_test_slide_names):
        print(f"Processing slide {i}...")

        # Find indices for the current slide in the test set
        SlIndex = np.where(test_slide_names == slide)[0]

        # Extract test data and labels for the current slide
        tst = np.array(test_features)[SlIndex]
        t_label = np.array(test_labels)[SlIndex]

        # Exclude data with the same slide number from the search space
        x_temple = np.delete(test_features, SlIndex, axis=0)
        y_temple = np.delete(test_labels, SlIndex, axis=0)

        # Train classifier on remaining data (excluding current slide samples)
        classifier.fit(x_temple, y_temple)

        # Predict labels for all test samples in the current slide at once
        y_pred = classifier.predict(tst)


        for idx, lbl in zip(SlIndex, y_pred):
            predicted_test_labels[idx] = lbl

    # Calculate accuracy
    acc = metrics.accuracy_score(test_labels, predicted_test_labels)

    return acc
