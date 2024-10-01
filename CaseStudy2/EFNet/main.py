
from CaseStudy2.EFNet.preprocessing import get_labels
from CaseStudy2.EFNet.dataBalancing import balancing
import numpy as np
from CaseStudy2.EFNet.EfficientNet import EF_model

# define variables

data_root = "C:/Users/kheir/Downloads/Shortcut_Learning/data_set"
total_slides= 100
selected_centers_dic = {"Johns Hopkins": total_slides, "Asterand": total_slides, "Indivumed": total_slides, "Roswell Park": total_slides}
cancers = ["Lung Squamous Cell Carcinoma", "Lung Adenocarcinoma"]
num_classes = len(cancers)


# read dataset
print("****************reading train data****************")

# this method preprocess the original patches (1024x1024) and generates the sub-patches (224x224)
data_images, data_cancer_labels,  data_center_labels, slidenames, filenames = get_labels(data_root)

# this method balances the data over both cancer types and data centers
reshaped_balanced_images, balanced_cancer_labels, balanced_center_labels, balanced_slidenames, filename = balancing(selected_centers_dic, data_images, data_center_labels, data_cancer_labels, cancers, slidenames, filenames)

# Manual dataSet Shuffeling ######################################################################
num_samples = data_images.shape[0]
# create a random permutation of indices
indices = np.random.permutation(num_samples)

# # the permutation to shuffle the dataset and labels
shuffled_reshaped_balanced_images = reshaped_balanced_images[indices]
shuffled_balanced_cancer_labels = balanced_cancer_labels[indices]
shuffled_balanced_center_labels = np.array(balanced_center_labels)[indices]



training_size = int(len(shuffled_balanced_center_labels) * 0.8)   # 80% of the array
validation_size = int(len(shuffled_balanced_center_labels) * 0.1)   # 10% of the array

# Split the array into three parts
# training
training_data = shuffled_reshaped_balanced_images[:training_size]
training_cancer_labels = shuffled_balanced_cancer_labels[:training_size]
training_center_labels = shuffled_balanced_center_labels[:training_size]


#validation
validation_data = shuffled_reshaped_balanced_images[training_size:training_size+validation_size]
validation_cancer_labels = shuffled_balanced_cancer_labels[training_size:training_size+validation_size]
validation_center_labels = shuffled_balanced_center_labels[training_size:training_size+validation_size]

#test
test_data = shuffled_reshaped_balanced_images[training_size+validation_size:]
test_cancer_labels = shuffled_balanced_cancer_labels[training_size+validation_size:]
test_center_labels = shuffled_balanced_center_labels[training_size+validation_size:]

######################################################################

# Model Training
trained_model = EF_model(num_classes, training_data, training_cancer_labels, validation_data, validation_cancer_labels)


