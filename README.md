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

1. MI.py
   This Python script computes the Mutual Information (MI) between cancer types and medical centers based on probability data from `probabiities.csv`

2. heatmap.py
3. This Python script generates a clustered heatmap using the data from `supplementary_material.csv.` The heatmap visualizes relationships between cancer types and medical centers, offering insight into the underlying patterns.





