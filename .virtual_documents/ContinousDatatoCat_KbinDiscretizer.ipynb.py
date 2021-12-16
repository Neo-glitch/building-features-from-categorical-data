from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


car_data = pd.read_csv("./datasets/auto-mpg.csv", na_values="?")

car_data.sample(8)


car_data.drop(columns = ["car name", "origin", "model year"], inplace = True)

car_data.dropna(inplace = True)
car_data.reset_index(inplace = True, drop = True)
car_data.sample(6)


# eda
fig, ax = plt.subplots(figsize = (12, 8))

plt.scatter(car_data["displacement"], car_data["horsepower"], color = "red")
ax.set(xlabel = "Engnie Displacement", ylabel = "HorsePower",
      title = "Displacement vs horsePower")
ax.grid()
plt.show()


# shows linear relationship


k_bins = KBinsDiscretizer(n_bins = 5,
                        encode = "ordinal",  # ret bin identifier encoded as int value
                        strategy="uniform") # bins have same width

k_bins


k_bins_array = k_bins.fit_transform(car_data[["displacement", "horsepower"]])

kbins_df = pd.DataFrame(data = k_bins_array, columns = ["bin_displacement", "bin_horsepower"])
kbins_df.sample(6)


# concat bin df with main df
car_data_k_bins = pd.concat([car_data, kbins_df], axis = 1)

car_data_k_bins.sample(6)


# find the unique bins for displacement
car_data_k_bins["bin_displacement"].unique()


# to check bin spacing of bin_displacment and bin_horsepower
displacement_edges = k_bins.bin_edges_[0]
horsepower_edges = k_bins.bin_edges_[1]

displacement_edges, horsepower_edges


car_data_k_bins["Comment"] = ''
car_data_k_bins.head()


car_data_k_bins.loc[car_data_k_bins["bin_displacement"] < 
                   car_data_k_bins["bin_horsepower"],
                   "Comment"] = "Efficient"


car_data_k_bins.loc[car_data_k_bins["bin_displacement"] > 
                   car_data_k_bins["bin_horsepower"],
                   "Comment"] = "Infficient"


car_data_k_bins.sample(10)


# data viz
fig, ax = plt.subplots(figsize = (10, 8))


categories = car_data_k_bins["Comment"].unique()

colors = {categories[0]: "green",
         categories[1]: "red",
         categories[2]: "blue"}

# plots scatter and colors points based on cat they belong to based on Comments col
ax.scatter(car_data_k_bins["displacement"],
          car_data_k_bins["horsepower"],
          c = car_data_k_bins["Comment"].apply(lambda x: colors[x]))

ax.set(xlabel = "Engine Displacement",
          ylabel = "HorsePower",
          title = "Displacement vs HorsePower")

ax.grid()
ax.set_xticks(displacement_edges)
ax.set_yticks(horsepower_edges)

plt.show()

















































































