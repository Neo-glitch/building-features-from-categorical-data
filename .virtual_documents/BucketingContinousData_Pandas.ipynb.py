import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid", color_codes = True)



age_df = pd.DataFrame(np.random.randint(0, 72, size = (120, 1)), columns = ["age"])  # dummy age data


age_df.head()


age_df.describe()


# bin edges
bins = [0, 18, 36, 54, 72]


# bins the numeric age data based on bin edges passed
age_df["range"] = pd.cut(age_df.age, bins)
age_df.head(8)


# bin counting(counting number of rows that fall under the bins)
age_df["range"].value_counts()


# data viz in bar chart using seaborn
sns.countplot(x = "range",
             data = age_df,
             palette = "hls")

plt.show();


































































