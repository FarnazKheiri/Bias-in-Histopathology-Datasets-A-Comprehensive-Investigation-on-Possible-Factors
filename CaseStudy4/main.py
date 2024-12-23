from CaseStudy2.EFNet.data_spilit import data_spilit
from CaseStudy2.EFNet.EfficientNet import EF_model
from CaseStudy4.noise_injection import to_gray
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

data_root = "C:/Users/kheir/Downloads/Shortcut_Learning/data_set"
total_slides= 100
selected_centers_dic = {"Johns Hopkins": total_slides, "Asterand": total_slides, "Indivumed": total_slides, "Roswell Park": total_slides}
cancers = ["Lung Squamous Cell Carcinoma", "Lung Adenocarcinoma"]
num_classes = len(cancers)

###################################################################### RGB Experiments

## Data loading
training_data,training_cancer_labels, training_center_labels, training_slide_names, validation_data, validation_cancer_labels, validation_center_labels,validation_slide_names, test_data, test_cancer_labels,test_center_labels,test_slide_names = data_spilit(data_root,selected_centers_dic,cancers)

##Model Training
rgb_trained_model = EF_model(num_classes, training_data, training_cancer_labels, validation_data, validation_cancer_labels)


layer_output_model = tf.keras.Model(inputs=rgb_trained_model.input, outputs=rgb_trained_model.layers[-2].output)
train_features = layer_output_model.predict(training_data)
test_features = layer_output_model.predict(test_data)

## KNN for cancer classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(train_features, training_cancer_labels)
y_pred = classifier.predict(test_features)
metrics.balanced_accuracy_score(np.argmax(y_pred, axis =1), np.argmax(test_cancer_labels, axis =1))

## KNN for center classification
# test_in_train center
k = 3
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(train_features, training_center_labels)
y_pred = classifier.predict(test_features)
metrics.balanced_accuracy_score(y_pred,test_center_labels)


###################################################################### Gray Experiments
## Data loading
training_data,training_cancer_labels, training_center_labels, _, validation_data, validation_cancer_labels, validation_center_labels,_, test_data, test_cancer_labels,test_center_labels,_ = data_spilit(data_root,selected_centers_dic,cancers)


## noise injection and gray conversion
gray_training_data = to_gray(training_data, th =0.7)
gray_validation_data = to_gray(validation_data, th =0.7)
gray_test_data = to_gray(test_data, th =0.7)

##Model Training
gray_trained_model = EF_model(num_classes, gray_training_data, training_cancer_labels, gray_validation_data, validation_cancer_labels)


# feature extraction
gray_layer_output_model = tf.keras.Model(inputs=gray_trained_model.input, outputs=gray_trained_model.layers[-2].output)
gray_train_features = gray_layer_output_model.predict(gray_training_data)
gray_test_features = gray_layer_output_model.predict(gray_test_data)


## KNN for cancer classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(gray_train_features, training_cancer_labels)
y_pred_gray = classifier.predict(gray_test_features)
metrics.balanced_accuracy_score(np.argmax(y_pred_gray, axis =1), np.argmax(test_cancer_labels, axis =1))

## KNN for center classification
# test_in_train center
k = 3
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(gray_train_features, training_center_labels)
y_pred_gray = classifier.predict(test_features)
metrics.balanced_accuracy_score(y_pred_gray,test_center_labels)


