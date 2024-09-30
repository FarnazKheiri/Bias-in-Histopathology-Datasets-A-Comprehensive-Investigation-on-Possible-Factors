# Bias-in-Histopathology-Datasets-A-Comprehensive-Investigation-on-Possible-Factors.

## KimiaNet Feature Filtering
This section contains a Python script to filter training, testing, or validation of TCGA features extracted by KimiaNet. This function filters the dataset for medical centers that contribute more than 40 slides and processes the corresponding features. The features are loaded from pre-saved pickle files for each slide.
* This final dataset includes patches originating from 38 acquisition sites with 29 cancer types.
  ### Code Description
  #### Parameters:
  data_set_type: Specifies the type of dataset to filter(train, test, validation)
  path: Path to the directory containing the KimiaNet feature files in .pickle format.
    
    
