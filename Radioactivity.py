import numpy as np
import matplotlib.pyplot as plt

N = 100000
num = [N]

while N>100:
    P = np.random.random(N)
    for p in P:
        if p>0.9:
            N=N-1
    num.append(N)

plt.plot(num)
plt.xlabel("Time (${t}_{0.5}$)")
plt.ylabel("Number of nucleotides")
plt.show()
