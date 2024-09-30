import pandas as pd
import seaborn as sns


# center_cancer_matrix.csv is same
mat_root_imbalanced = "./supplemetary_material.csv"

data = pd.read_csv(mat_root_imbalanced, header=[0], index_col=[0])
df_data = pd.DataFrame(data)

sns.set(font_scale=0.8)
sns.clustermap(df_data, cmap="YlGnBu", figsize=(8, 10), method="average", metric="correlation",
               z_score=(0, 1), cbar_pos=(-0.04, .6, .03, .2))


