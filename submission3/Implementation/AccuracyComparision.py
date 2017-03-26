import matplotlib.pyplot as plt
import numpy as np

n = np.array([10, 15, 20, 25])

t1 = np.array([94.62,95.72,95.80,95.97])
t2 = np.array([95.20,95.59,96.03,96.48])
t3 = np.array([92.35,92.97,93.23,96.48])
t4 = np.array([94.66,95.08,96.07,96.10])

#legend = []
fig = plt.figure(1)

plt.subplot(211)
plt.plot(n,t1,color="blue")
plt.plot(n,t2,color="red")
plt.plot(n,t3,color="green")
plt.plot(n,t4,color="black")
plt.legend(['Hidden Layers: 2', 'Hidden Layers: 3', 'Hidden Layers: 4', 'Hidden Layers: 5'])
plt.suptitle('Accuracy of different layers on increasing number of epochs and 500 neurons')
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy in %')

n = np.array([1000, 500, 100])
t1 = np.array([83.77,87.08,92.07])
t2 = np.array([90.97,93.13,94.62])
t3 = np.array([82.26,85.16,92.35])
t4 = np.array([78.12,87.50,94.66])
plt.subplot(212)
plt.plot(n,t1,color="blue")
plt.plot(n,t2,color="red")
plt.plot(n,t3,color="green")
plt.plot(n,t4,color="black")
plt.legend(['Hidden Layers: 2', 'Hidden Layers: 3', 'Hidden Layers: 4', 'Hidden Layers: 5'])
plt.suptitle('Accuracy of different layers on decreasing number of batch size and 100 neurons')
plt.xlabel('Batch size')
plt.ylabel('Accuracy in %')

plt.show()
fig.savefig('AccuracyComparision.jpg')
