
from preprocessing import get_labels
from dataBalancing import balancing



# define data root
data_root = "C:/Users/kheir/Downloads/Shortcut_Learning/data_set"
train_filtered_data_dic = {"University of Pittsburgh": 100, "Johns Hopkins": 100, "International Genomics Consortium": 100}

# read dataset
print("****************reading train data****************")

# this method preprocess the original patches (1024x1024) and generates the sub-patches (224x224)
data_images, data_cancer_labels,  data_center_labels, slidenames, filenames = get_labels(data_root)

