import os
import pandas as pd
import pickle


def KimiaNet_Features_Filtering(set_type="train"):
  feature = list()
  y_feature_center = list()
  y_feature_cancer = list()

  path = "/content/drive/MyDrive/KimiaNet Features/AllKimiaPatches"
  dataList = os.listdir(path)

  # Load the patch information
  data = pd.read_csv("patch_info.csv")
  df_data = pd.DataFrame(data=data)

  # Filter the dataframe by the given set type ("train" or others)
  first = data.loc[df_data["set"] == set_type]

  for mc in set(first["medical_center"]):
    med = first.loc[first["medical_center"].isin([mc])]
    med_count = len(med)

    if med_count > 40:
      for i in range(len(med)):
        second_filtering = med.iloc[i]
        name = second_filtering["slide_name"]
        label_mc = second_filtering["medical_center"]
        label_dis = second_filtering["disease_type"]

        # Construct the filename and path
        filename = name.replace(".svs", "_KimiaNet_features_dict.pickle")
        directory = os.path.join(path, filename)

        # If the file is not found, continue to the next one
        if dataList.count(filename) == 0:
          continue

        # Load the features
        with open(directory, 'rb') as f:
          x = pickle.load(f)

        # Append the data to the lists
        for key, value in x.items():
          feature.append(value)
          y_feature_center.append(label_mc)
          y_feature_cancer.append(label_dis)

  return feature, y_feature_center, y_feature_cancer


# Example usage
x_train, y_train_center, y_train_patch_disease_type, train_patch_keys = KimiaNet_Features_Filtering("train")

x_data, x_data_center, x_data_cancer =  KimiaFeatures_dataFilitring (data_set_type,path)