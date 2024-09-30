import pandas as pd
import numpy as np


file_path = 'probabilities.csv'
data = pd.read_csv(file_path)

# Extract the columns of interest
P_cancer = data['P(cancer)'].values
P_center = data['P(center)'].values
P_center_cancer = data['P(cancer,center)'].values


MI = 0


for i in range(len(P_cancer)):
    # Calculate P(cancer, center) / (P(cancer) * P(center))
    if P_center_cancer[i] > 0 and P_cancer[i] > 0 and P_center[i] > 0:  # Avoid log(0)
        MI += P_center_cancer[i] * np.log(P_center_cancer[i] / (P_cancer[i] * P_center[i]))


print(f"Mutual Information (MI) between cancer and center: {MI}")
