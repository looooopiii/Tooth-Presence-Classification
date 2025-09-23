import pandas as pd

df_lower = pd.read_csv("/home/user/lzhou/week8/data_augmentation/Aug_data/manifest_lower.csv")
df_upper = pd.read_csv("/home/user/lzhou/week8/data_augmentation/Aug_data/manifest_upper.csv")

df_all = pd.concat([df_lower, df_upper], ignore_index=True)
df_all.to_csv("/home/user/lzhou/week8/data_augmentation/Aug_data/manifest_all.csv", index=False)

print("Merged manifest saved to manifest_all.csv")