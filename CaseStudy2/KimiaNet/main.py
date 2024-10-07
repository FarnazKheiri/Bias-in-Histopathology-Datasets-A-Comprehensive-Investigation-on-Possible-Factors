from KimiaNet_Featute_Filtering import KimiaNet_Features_Filtering
from CaseStudy2.test_in_test import test_in_test
from CaseStudy2.test_in_train import test_in_train
from CaseStudy2.excluded_test_in_test import excluded_test_in_test


############################################################load festures of KimiaNet
x_test, y_test_center, y_test_cancer, test_slide_names = KimiaNet_Features_Filtering("test")
x_train, y_train_center, y_train_cancer, train_slide_names = KimiaNet_Features_Filtering("train")

############################################################# test_in_test search
#cancer classification
test_in_test_cancer_acc=test_in_test(k=3, features=x_test, labels=y_test_cancer)
print("test_in_test_cancer_acc" + "is" + test_in_test_cancer_acc)

# center classification
test_in_test_center_acc = test_in_test(k=3, features=x_test, labels=y_test_cancer)
print("test_in_test_center_acc" + "is" + test_in_test_center_acc)

############################################################# test_in_train search
#cancer classification
test_in_train_cancer_acc = test_in_train(k=3, test_features=x_test, test_labels=y_test_cancer, train_features=x_train, train_labels=y_train_cancer)
print("test_in_train_cancer_acc" + "is" + test_in_train_cancer_acc)

#center classification
test_in_train_center_acc = test_in_train(k=3, test_features=x_test, test_labels=y_test_center, train_features=x_train, train_labels=y_train_center)
print("test_in_train_center_acc" + "is" + test_in_train_center_acc)

############################################################# test_in_test search with exclusion condition
#cancer classification
ex_test_in_test_cancer_acc = excluded_test_in_test(k=3,test_features=x_test, test_labels=y_test_cancer,test_slide_names=test_slide_names)
print("ex_test_in_test_cancer_acc" + "is" + ex_test_in_test_cancer_acc)
#center classification
ex_test_in_test_center_acc = excluded_test_in_test(k=3, test_features=x_test, test_labels=y_test_center,test_slide_names=test_slide_names)
print("ex_test_in_test_center_acc" + "is" + ex_test_in_test_center_acc)
