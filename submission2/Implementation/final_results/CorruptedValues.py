import matplotlib.pyplot as plt
import numpy as np

n = np.array([5, 8, 15, 22, 30])

t1 = np.array([6.17, 7.77, 10.71, 12.73, 14.94])

t2 = np.array([3.20, 3.52, 4.71, 7.16, 10.51])

fig = plt.figure(1)

plt.subplot(211)
plt.plot(n,t1)
plt.plot(n,t2)
plt.legend(['PPCA(EM)', 'RPCA'])
plt.suptitle('Comparision of PPCA(EM) and RPCA for Corrupted Values values')
plt.xlabel('% of data as Corruptedvalues')
plt.ylabel('Error in %')
plt.subplot(212)
t1 = np.array([0.73, 0.897, 1.24, 1.52, 1.78])
t2 = np.array([2.95, 3.18, 3.75, 4.76, 7.85])
plt.plot(n,t1)
plt.plot(n,t2)
plt.legend(['PPCA(EM)', 'RPCA'])
plt.suptitle('Comparision of PPCA(EM) and RPCA for Missing values')
plt.xlabel('% of data as MissingValues')
plt.ylabel('Error in %')
plt.show()
fig.savefig('ErrorComparision.jpg')
