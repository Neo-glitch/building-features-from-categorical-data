import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm


car_data = pd.read_csv("./datasets/auto-mpg.csv", na_values="?")

car_data.sample(8)


# using just mpg and horsepower for work
car_data = car_data[["mpg", "horsepower"]]

car_data.dropna(inplace = True)
car_data.reset_index(inplace = True, drop = True)

car_data.sample(6)


# horse power is inversely corr with mpg
car_data.corr()


# cretes 4 buckets that are equally spaced, n.b 3 bin edges = 4 buckets
_, bin_edges = np.histogram(car_data["horsepower"], 3)


# we bin edge(firt bucket is hp values of 46 - 107.33333)
bin_edges


# coverts equally spaced bins to cat values
hp_cat = np.digitize(car_data.horsepower, bin_edges, True)

hp_cat[:30]


# creat col in df for this cat values
car_data["hp_cat"] = hp_cat

car_data.sample(6)


# no need for horsePower col again
car_data.drop(columns = ["horsepower"], inplace = True)


# group data by cat levels
car_data_grouped = car_data.groupby("hp_cat")
car_data_grouped.head()


# mean mpg for all bins cat
car_data_grouped.mean()


# mean of cat means
car_data_grouped.mean().mean()


mod = ols("mpg ~ C(hp_cat, Poly)", data = car_data)

result = mod.fit()
result.summary()


from patsy.contrasts import Poly


levels = [0, 1, 2, 3]


contrast_with_int = Poly().code_with_intercept(levels)

contrast_with_int


contrast_without_int = Poly().code_without_intercept(levels)

contrast_without_int


car_data_contrast = contrast_without_int.matrix[car_data.hp_cat - 0, :]

car_data_contrast[:6]


# df with encoded values
car_data_contrast_df = pd.DataFrame(car_data_contrast,
                                    columns = ["linear", "quadratic", "cubic"])

car_data_contrast_df.sample(7)


# concat encoded df with main df
car_data_enc = pd.concat([car_data, car_data_contrast_df], axis = "columns")

car_data_enc.sample(6)


X = car_data_enc.drop(columns = ["mpg", "hp_cat"], axis = 1)
y = car_data_enc["mpg"]


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept = True)
linear_model.fit(X, y)

print("Training_score: ", linear_model.score(X, y))


linear_model.coef_


linear_model.intercept_































































