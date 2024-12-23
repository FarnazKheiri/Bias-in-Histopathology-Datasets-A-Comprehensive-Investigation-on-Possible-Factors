from CaseStudy2.EFNet.preprocessing import get_labels
from CaseStudy2.EFNet.dataBalancing import balancing
import numpy as np

# define variables



def data_spilit(data_root,selected_centers_dic,cancers):
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
    shuffled_balanced_slidenames = np.array(balanced_slidenames)[indices]


    training_size = int(len(shuffled_balanced_center_labels) * 0.8)   # 80% of the array
    validation_size = int(len(shuffled_balanced_center_labels) * 0.1)   # 10% of the array

    # Split the array into three parts
    # training
    training_data = shuffled_reshaped_balanced_images[:training_size]
    training_cancer_labels = shuffled_balanced_cancer_labels[:training_size]
    training_center_labels = shuffled_balanced_center_labels[:training_size]
    training_slide_names = shuffled_balanced_slidenames[:training_size]


    #validation
    validation_data = shuffled_reshaped_balanced_images[training_size:training_size+validation_size]
    validation_cancer_labels = shuffled_balanced_cancer_labels[training_size:training_size+validation_size]
    validation_center_labels = shuffled_balanced_center_labels[training_size:training_size+validation_size]
    validation_slide_names = shuffled_balanced_slidenames[training_size:training_size+validation_size]

    #test
    test_data = shuffled_reshaped_balanced_images[training_size+validation_size:]
    test_cancer_labels = shuffled_balanced_cancer_labels[training_size+validation_size:]
    test_center_labels = shuffled_balanced_center_labels[training_size+validation_size:]
    test_slide_names = shuffled_balanced_slidenames[training_size:training_size+validation_size]

    return training_data,training_cancer_labels, training_center_labels, training_slide_names, validation_data, validation_cancer_labels, validation_center_labels,validation_slide_names, test_data, test_cancer_labels,test_center_labels,test_slide_names
