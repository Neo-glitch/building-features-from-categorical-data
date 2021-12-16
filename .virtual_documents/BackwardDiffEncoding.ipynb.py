import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

import category_encoders as ce


iris_data = pd.read_csv("./datasets/iris.csv")
iris_data.sample(10)


iris_data.drop(columns = ["sepal_length", "sepal_width", "petal_width"], inplace = True)

iris_data.sample(5)


iris_data.describe()


# mean petal length value by species
iris_data.groupby("Species").mean()


# diff in mean value in each cat by comparing it with mean value of previous cat
iris_data.groupby("Species").mean().diff()


# backward diff encoding
mod = ols("petal_length ~ C(Species, Diff)",  # diff is for backward diff encoding
         data = iris_data)  

res = mod.fit()
res.summary()


encoder = ce.BackwardDifferenceEncoder(cols = ["Species"])  # col to be encoded
encoder


species_encoded = encoder.fit_transform(iris_data)

species_encoded.sample(5)


encoded_iris = pd.concat([iris_data["Species"], species_encoded], axis = 1)

encoded_iris.sample(6)


X = encoded_iris.drop(columns = ["Species", "petal_length"])

y = encoded_iris.petal_length


# since intercept added by default for encoded data, fit linear model without intercept
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept = False)

linear_model.fit(X, y)
print("Training score: ", linear_model.score(X, y))


linear_model.coef_







































































































































