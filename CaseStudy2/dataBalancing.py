import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random

def balancing(filtered_data_dic,images, center_labels,cancer_labels, cancers, slidenames, filenames):
    balanced_images = list()
    balanced_center_labels = list()
    balanced_cancer_labels = list()
    balanced_slidenames = list()
    balanced_filenames = list()
    for key, value in filtered_data_dic.items():
        print(key)
        for cancer in cancers:
            idx = np.where((np.array(center_labels) == key) & (np.array(cancer_labels) == cancer))[0]
            random_numbers = random.sample(range(0, len(idx)), value*25)
            filtered_images  = [images[i] for i in np.array(idx)[random_numbers]]
            balanced_images.extend(filtered_images)
            balanced_center_labels.extend(list(np.array(center_labels)[np.array(idx)[random_numbers]]))
            balanced_cancer_labels.extend(list(np.array(cancer_labels)[np.array(idx)[random_numbers]]))
            balanced_slidenames.extend(list(np.array(slidenames)[np.array(idx)[random_numbers]]))
            balanced_filenames.extend(list(np.array(filenames)[np.array(idx)[random_numbers]]))
#             pdb.set_trace()
    ### reshape images from (224,224,3) to (N, 224,224,3)
    reshaped_balanced_images = np.stack(balanced_images)
    ### Convert cancer labels from "string" to "num"
    label_encoder = LabelEncoder()
    numeric_cancer_labels = label_encoder.fit_transform(balanced_cancer_labels)
    # Convert numeric labels to one-hot encoded labels using OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False)
    cancer_encoded_balanced_labels = ohe.fit_transform(numeric_cancer_labels.reshape(-1, 1))
    ### Convert cancer labels from "string" to "num"
    numeric_center_labels = label_encoder.fit_transform(balanced_center_labels)
    # Convert numeric labels to one-hot encoded labels using OneHotEncoder
    cen = OneHotEncoder(sparse_output=False)
    center_encoded_balanced_labels = cen.fit_transform(numeric_center_labels.reshape(-1, 1))
    # Data Shuffeling
#     pdb.set_trace()
    return reshaped_balanced_images, cancer_encoded_balanced_labels, balanced_center_labels, balanced_slidenames, balanced_filenames
