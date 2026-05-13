"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete

y_des = np.load("2.npz")["arr_0"].T
y_des -= y_des[:, -1][:, None]
print(y_des.shape)
# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=10,y0=y_des[:,0])

f_demo = dmp.imitate_path(y_des=y_des, plot=False)
y_predicted, dy_predicted, ddy_predicted, f_learn = dmp.rollout()
plt.figure(1, figsize=(6, 6))

plt.plot(y_predicted[:, 0], y_predicted[:, 1], "b", lw=2)
plt.title("DMP system - draw number 2")

plt.axis("equal")
plt.show()
