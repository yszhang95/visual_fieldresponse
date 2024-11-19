import numpy as np
xscipy = np.loadtxt('scipy_quad_cpu.txt')
xtorch = np.loadtxt('gaus_quad_4points_gpu.txt')
print(xscipy.shape)
print(xtorch.shape)
print(np.sum((xscipy-xtorch)[:,0:9]))
d = np.abs(xscipy[:,9]-xtorch[:,9])
print(np.min(d), np.max(d))
print(np.min(xscipy[:,10]), np.max(xscipy[:,10]))
ytorch = xtorch[:,9]
yscipy = xscipy[:,9]
m = ytorch > 1E-5

print(np.min(xscipy[:,10]), np.max(xscipy[:,10]))
print(np.min(ytorch[m]))
print(np.max(
    np.abs(ytorch[m]-yscipy[m])/ytorch[m]
))

print(yscipy[np.argmax(yscipy)], d[np.argmax(yscipy)])
print(yscipy[np.argmax(d)], d[np.argmax(d)])
