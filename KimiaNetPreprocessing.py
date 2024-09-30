
import os
import pickle
import pandas as pd





path = "/content/drive/MyDrive/KimiaNet Features/AllKimiaPatches"

data_set_type = "train"



def KimiaFeatures_dataFilitring(data_set_type,path):

  x_data = list()
  x_data_center = list()
  x_data_cancer = list()

  dataList = os.listdir(path)

  data = pd.read_csv("patch_info.csv")
  df_data = pd.DataFrame(data=data)

  # data_set_type must be either train,test or validation
  first_filter = data.loc[df_data["set"]==data_set_type]

  for mc in set(first_filter["medical_center"]):
      med = first_filter.loc[first_filter["medical_center"].isin([mc])]
      med_count = len(med)
      if (med_count > 40):
          for i in range(len(med)):
            second_filter =med.iloc[i]
            name = second_filter["slide_name"]
            label_mc = second_filter["medical_center"]
            label_dis = second_filter["disease_type"]
            filename = name.replace(".svs","_KimiaNet_features_dict.pickle")
            directory = path + "/" + filename
            if (dataList.count(filename) == 0):
              continue
            with open(directory, 'rb') as f:
              x = pickle.load(f)
            for key, value in x.items():
              x_data.append(value)
              x_data_center.append(label_mc)
              x_data_cancer.append(label_dis)
  return x_data, x_data_center, x_data_cancer


x_data, x_data_center, x_data_cancer =  KimiaFeatures_dataFilitring (data_set_type,path)