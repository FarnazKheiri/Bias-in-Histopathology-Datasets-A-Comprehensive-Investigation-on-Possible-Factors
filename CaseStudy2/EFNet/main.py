
from CaseStudy2.EFNet.EfficientNet import EF_model
from CaseStudy2.test_in_test import test_in_test
from CaseStudy2.test_in_train import test_in_train
from CaseStudy2.excluded_test_in_test import excluded_test_in_test
from CaseStudy2.EFNet.data_spilit import data_spilit
import tensorflow as tf

data_root = "./data_set"
total_slides= 100
selected_centers_dic = {"Johns Hopkins": total_slides, "Asterand": total_slides, "Indivumed": total_slides, "Roswell Park": total_slides}
cancers = ["Lung Squamous Cell Carcinoma", "Lung Adenocarcinoma"]
num_classes = len(cancers)

###################################################################### Data Prepration
training_data,training_cancer_labels, training_center_labels, training_slide_names, validation_data, validation_cancer_labels, validation_center_labels,validation_slide_names, test_data, test_cancer_labels,test_center_labels,test_slide_names = data_spilit(data_root,selected_centers_dic,cancers)


###################################################################### Model Training
trained_model = EF_model(num_classes, training_data, training_cancer_labels, validation_data, validation_cancer_labels)

####################################################################### Feature Extraction
layer_output_model = tf.keras.Model(inputs=trained_model.input, outputs=trained_model.layers[-2].output)
train_features = layer_output_model.predict(training_data)
test_features = layer_output_model.predict(test_data)

############################################################# test_in_test search
#cancer classification
test_in_test_cancer_acc=test_in_test(k=3, features=test_features, labels=test_cancer_labels)
print("test_in_test_cancer_acc" + "is" + test_in_test_cancer_acc)

# center classification
test_in_test_center_acc = test_in_test(k=3, features=test_features, labels=test_center_labels)
print("test_in_test_center_acc" + "is" + test_in_test_center_acc)

############################################################# test_in_train search
#cancer classification
test_in_train_cancer_acc = test_in_train(k=3, test_features=test_features, test_labels=test_cancer_labels, train_features=train_features, train_labels=training_cancer_labels)
print("test_in_train_cancer_acc" + "is" + test_in_train_cancer_acc)

#center classification
test_in_train_center_acc = test_in_train(k=3, test_features=test_features, test_labels=test_center_labels, train_features=train_features, train_labels=training_center_labels)
print("test_in_train_center_acc" + "is" + test_in_train_center_acc)

############################################################# test_in_test search with exclusion condition
#cancer classification
ex_test_in_test_cancer_acc = excluded_test_in_test(k=3,test_features=test_features, test_labels=test_cancer_labels,test_slide_names=test_slide_names)
print("ex_test_in_test_cancer_acc" + "is" + ex_test_in_test_cancer_acc)
#center classification
ex_test_in_test_center_acc = excluded_test_in_test(k=3, test_features=test_features, test_labels=test_center_labels,test_slide_names=test_slide_names)
print("ex_test_in_test_center_acc" + "is" + ex_test_in_test_center_acc)

