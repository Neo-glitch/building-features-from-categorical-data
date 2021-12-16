import pandas as pd
import numpy as np
from statsmodels.formula.api import ols


car_data = pd.read_csv("./datasets/auto-mpg.csv", na_values="?")

car_data.sample(8)


# get just two cols to use as X and y, cylinder col is the col that will be helmert encoded
car_data = car_data[["mpg", "cylinders"]]


car_data.dropna(inplace = True)


car_data.sample(6)


car_data.cylinders.unique()  # since unique num ain't continous, we can say it a cat data


# sort cars based on numner of cylinders
car_data.sort_values(by = ["cylinders"], inplace = True)
car_data.reset_index(inplace=True, drop=True)
car_data.head(10)


car_data.mean()


# calc avg mpg for each cyclinder cat
car_data_grouped = car_data.groupby(by = ["cylinders"]).mean()

car_data_grouped


# get means of all cat(i.e mean of grouped means(mean mpg))
car_data_grouped["mpg"].mean()


# calc coef of cat 4 using helmer coding manually, divide by 2 since working with 2 cats
coefficient_cylinder_4 = (car_data_grouped.loc[4]["mpg"] - car_data_grouped.loc[3]["mpg"]) / 2

coefficient_cylinder_4


mean_34 = (car_data_grouped.loc[3]["mpg"] + car_data_grouped.loc[4]["mpg"]) / 2 

coefficient_cylinder_5 = (car_data_grouped.loc[5]["mpg"] - mean_34) / 3 # 3 since 3 cat involved
coefficient_cylinder_5


mod = ols("mpg ~ C(cylinders, Helmert)", data = car_data)

res = mod.fit()
res.summary()

# intercept is same as cat means


import category_encoders as ce


ce_helmert = ce.HelmertEncoder(cols = ["cylinders"])
car_he = ce_helmert.fit_transform(car_data)

car_he.sample(6)


X = car_he.drop(columns = ["mpg"], axis = 1)
y = car_he.mpg


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept = False)
linear_model.fit(X, y)

print("Training_score: ", linear_model.score(X, y))


linear_model.coef_













































































































































