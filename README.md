# **Bias in Histopathology Datasets: A Comprehensive Investigation on Possible Factors**

## **KimiaNet Feature Filtering**

This section presents a Python script designed to filter and process TCGA features extracted by KimiaNet for **training**, **testing**, or **validation** purposes. The script filters the dataset based on the number of slides contributed by each medical center and processes only those centers that contribute more than **40 slides**. The features for each slide are loaded from pre-saved **pickle** files.

- The final dataset includes **patches** originating from **38 acquisition sites**, covering **29 different cancer types**.

### **Description**

#### **Parameters:**
- **`data_set_type`**: Specifies the type of dataset to filter (`train`, `test`, or `validation`).
- **`path`**: The directory path containing the KimiaNet feature files in `.pickle` format.

#### **Process Overview:**
- The script reads a CSV file (`patch_info.csv`) containing metadata about each patch, including details such as the **medical center**, **cancer type**, and **slide name**.
- The dataset is filtered based on the type of data (train, test, or validation), ensuring that only slides from medical centers with more than 40 contributions are included.

## **CaseStudy1: Mutual Information and Heatmap Analysis**

This case study focuses on calculating the Mutual Information (MI) between cancer types and medical centers and visualizing the data using a heatmap to highlight patterns or relationships between various factors.
### **Description**

#### 1. MI.py ####
   This Python script computes the Mutual Information (MI) between cancer types and medical centers based on probability data from `probabiities.csv`. The probabilities in this file are calculated using the matrix provided in the `supplementary_material.csv`.

#### 2. heatmap.py ####
   This Python script generates a clustered heatmap using the data from `supplementary_material.csv`. The heatmap visualizes relationships between cancer types and medical centers, offering insight into the underlying patterns.


## **CaseStudy2: Investigating the Impact of Patching**

### Key Files:
- **EFNet/**
  - `EfficientNet.py`: Script to finetune the EfficientNet model for patch-based classification.
  - `dataBalancing.py`: Methods for balancing data.
  - `main.py`: Entry point for executing the experiments using EfficientNet.
  - `preprocessing.py`: Preprocessing steps applied to patches to create subpatches aimed at adopting the input size for EfficientNet.

- **KimiaNet/**
  - `main.py`: Main script to run the KimiaNet model and related experiments.

- **test_in_test.py**: Code for **Test Case 1**, where the k-NN classifier uses the test set as its own search space for classification, without excluding co-slide patches.

- **test_in_train.py**: Code for **Test Case 2**, where the k-NN classifier uses the training set as the search space for test samples.
- **excluded_test_in_test.py**: Code for **Test Case 3**, where co-slide patches are excluded during the k-NN classification process.



