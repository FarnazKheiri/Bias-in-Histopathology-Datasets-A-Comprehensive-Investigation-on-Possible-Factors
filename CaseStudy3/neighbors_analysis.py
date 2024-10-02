from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


def neighbor_analysis(k,test_features,test_cancer_labels,test_center_labels,test_slide_names):
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Create an empty DataFrame for storing neighbors
    neighbors_mat = pd.DataFrame(
        np.ones((len(test_features), k)) * -1
    )

    predicted_test_labels = [None] * len(test_features)
    unique_test_slide_name = np.unique(test_slide_names)


    for i, slide in enumerate(unique_test_slide_name):
        print(f"Processing slide {i}...")

        # Find indices for the current slide in the test set
        SlIndex = np.where(test_slide_names == slide)[0]

        # Extract test data and labels for the current slide
        tst = np.array(test_features)[SlIndex]
        t_label = np.array(test_center_labels)[SlIndex]


        x_temple = np.delete(test_features, SlIndex, axis=0)
        center_temple = np.delete(y_test_center, SlIndex, axis=0)
        cancer_temple = np.delete(test_cancer_labels, SlIndex, axis=0)


        classifier.fit(x_temple, center_temple)

        # Predict labels for all test samples in the current slide at once
        center_pred = classifier.predict(tst)

        # Find neighbors for the test samples
        neighbors = classifier.kneighbors(tst, return_distance=False)

        # Update the patch neighbors matrix for correct predictions
        for idx in range(len(SlIndex)):
            neighbors_mat.iloc[SlIndex[idx]] = cancer_temple[neighbors[idx]]

        # Update predicted test labels for all test samples in the current slide
        for idx, lbl in zip(SlIndex, center_pred):
            predicted_test_labels[idx] = lbl



    return neighbors_mat