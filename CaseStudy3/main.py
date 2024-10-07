from KimiaNet_Featute_Filtering import KimiaNet_Features_Filtering
from neighbors_analysis import neighbor_analysis
import numpy as np

############################################################load festures of KimiaNet

x_test, y_test_center, y_test_cancer, test_slide_names = KimiaNet_Features_Filtering("test")
k =3


neighbors = neighbor_analysis(k,x_test, y_test_cancer, y_test_center, test_slide_names)

# save the centers' names of neighbors_with_same_center
temp_cancers_each_row = {}
count_number_of_neighbors_with_same_cancer = []
names_neighbors_with_same_cancer = []


# define an array to save number of neighbors with the same cancer type for each patch
number_of_neighbors_with_same_cancer = {}
temp_centers_each_row = {}
center_neighbors_cancers = {}


# get the rows that are not -1. (correctly classified "center classification")
row_sums = np.sum(neighbors, axis=1)
indices = np.where(row_sums > 0)[0]
rows_with_correct_classification = neighbors[indices]


# number_of_neighbors_with_same_cancer = np.ones((1, len(indices)))
for idx, value in zip(indices, rows_with_correct_classification):
  neighbors_with_same_cancer = np.where(np.array(y_test_cancer)[value.astype(int)] == y_test_cancer[idx])

  # number of neighbors belonging to one sample that have same cancer label that the sample has
  number_of_neighbors_with_same_cancer[idx] = len(neighbors_with_same_cancer[0])

  if number_of_neighbors_with_same_cancer[idx] > 0:
    # to see, which cancer types mostly are participate in bias results
    temp_cancers_each_row[idx] = set(np.array(y_test_cancer)[value.astype(int)[neighbors_with_same_cancer[0]]])

    count_number_of_neighbors_with_same_cancer.append(number_of_neighbors_with_same_cancer[idx])

    names_neighbors_with_same_cancer.append(set(np.array(y_test_cancer)[value.astype(int)[neighbors_with_same_cancer[0]]]))

    center_neighbors_cancers[y_test_center[idx] + str(idx)] = temp_cancers_each_row[idx]

