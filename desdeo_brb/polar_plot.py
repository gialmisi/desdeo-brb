import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas.plotting import parallel_coordinates


candidates = np.array([[0.5, 0.75, 0.25],
                       [0.22, 0.55, 0.82],
                       [0.11, 0.22, 0.33]])
obj_names = ["INCOME", "CO2", "HSI"]

df = pandas.DataFrame(data=candidates,
                      index=[f"Candidate {i+1}" for i in range(len(candidates))],
                      columns=obj_names).reset_index()


parallel_coordinates(df, "index")
plt.show()
#data = np.zeros((candidates.shape[0], candidates.shape[1]+1))
#data[:, :-1] = candidates
#data[:, -1] = candidates[:, 0]
#print(data)
#
#n_vars = candidates.shape[1]
#
#angles = [n/n_vars*2*np.pi for n in range(n_vars)]
#angles += angles[:1]
#print(angles)
#
#ax = plt.subplot(111, polar=True)
#
#ax.set_theta_offset(np.pi / 2)
#ax.set_theta_direction(-1)
#
#plt.xticks(angles[:-1], obj_names)
#
#ax.set_rlabel_position(0)
#plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], color="grey", size=7)
#plt.ylim(0, 1)
#
#ax.plot(angles, data[0], linewidth=1, linestyle="solid", label="Candidate 1")
#ax.fill(angles, data[0], "b", alpha=0.1)
#ax.plot(angles, data[1], linewidth=1, linestyle="solid", label="Candidate 2")
#ax.fill(angles, data[1], "r", alpha=0.1)
#
#plt.show()
