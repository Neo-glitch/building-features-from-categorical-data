import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# lib that does feature hashing, and scales well to large data and good for online learning
from sklearn.feature_extraction import FeatureHasher   


hasher = FeatureHasher(n_features = 5, input_type = "dict")


random_dict = hasher.fit_transform([
    {"a":1, "b": 2},
    {"a": 0, "c": 5}])


random_dict # 2x5 means 2 records(dict here) rep using 5 features


random_dict.toarray()


# feature hasher on pairs(tuples)
hasher_pair = FeatureHasher(n_features = 5, input_type = "pair")


random_pair= hasher_pair.fit_transform([
    [("a", 1), ("b", 2)],
    [("a", 0), ("c", 5)]
])


random_pair


# feature hasher on text(normal use of this)

text = ["This is me fool",
       "Hi, my name is sman",
       "The best team in england is chelsea",
       "I think Nigeria is cursed"]


hasher = FeatureHasher(n_features = 8, input_type = "string")


hashed_features = hasher.fit_transform(text)


hashed_features


ozone_reading = pd.read_csv("./datasets/ozone_reading.csv")

ozone_reading.sample(6)


ozone_reading.Month.unique()


ozone_reading.describe()


ozone_reading_correlation = ozone_reading.corr()

ozone_reading_correlation

fig, ax = plt.subplots(figsize = (10, 8))
sns.heatmap(ozone_reading_correlation,
           annot = True,
           cmap = "viridis")

plt.show();


# feature hasher
fh = FeatureHasher(n_features = 4, input_type = "string")


hashed_features = fh.fit_transform(ozone_reading["Month"])


hashed_features = hashed_features.toarray()

hashed_features[:8]


# cat to rep the hashed values
hashed_categories = ["month_hash_0", "month_hash_1", "month_hash_2", "month_hash_3"]


hashed_df = pd.DataFrame(hashed_features, columns = hashed_categories, dtype = np.int)

hashed_df.sample(7)


# concat original df and hashed df
hashed_ozone_reading = pd.concat([ozone_reading, hashed_df], axis = 1)

hashed_ozone_reading.sample(5)


# drop oringinal month column, no use again
hashed_ozone_reading.drop(columns = ["Month"], inplace = True)


X = hashed_ozone_reading.drop("ozone_reading", axis = 1)
y = hashed_ozone_reading.ozone_reading


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X, y)

print("Training score: ", linear_model.score(X, y))








































































