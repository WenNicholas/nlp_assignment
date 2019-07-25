# -*- coding:utf-8 -*-
"""
参考优秀同学代码，链接：https://mp.weixin.qq.com/s/HZQm6qtQ9QXasoHzkjag2g
"""
import random
import matplotlib.pyplot as plt
# 随机生成20个点
latitudes = [random.randint(-100, 100) for _ in range(20)]
longitude = [random.randint(-100, 100) for _ in range(20)]
chosen_p = (-50, 10) # 起点
plt.scatter(latitudes, longitude)
# plt.show()

import pandas as pd
import numpy as np
import math
import time

point_list = [(x, y) for x, y in zip(latitudes, longitude)]
point_list.insert(0, chosen_p)
point_array = np.array(point_list)

# 距离矩阵
dist = np.zeros((point_array.shape[0], point_array.shape[0]))
for i in range(point_array.shape[0]):
    for j in range(point_array.shape[0]):
        dist[i, j] = math.sqrt(np.sum((point_array[i, :] - point_array[j, :]) ** 2))

"""
N:计数
s:二进制表示，遍历过得城市对应位为0，未遍历为1
dp:动态规划的距离数组
dist：目的地间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
path:记录下一个应该到达的城市
"""

N = point_array.shape[0]
path = np.ones((2 ** N - 1, N), dtype=np.int)
dp = np.ones((2 ** N - 1, N)) * -1


# 代码的核心
def TSP(s, init):
    if dp[s][init] != -1:
        return dp[s][init]

    if s == 0:  # 成立代表遍历结束
        return dist[0][init]

    sumpath = float('inf')

    for i in range(N):
        if s & (1 << i):  # 判断是否遍历过，未遍历则执行
            m = TSP(s & (~ (1 << i)), i) + dist[i][init]  # s & (~ (1 << i))让遍历过的点的相应位置变0
            if m < sumpath:
                sumpath = m
                path[s][init] = i
    dp[s][init] = sumpath
    return dp[s][init]


init_point = 0
s = 0
for i in range(1, N):
    s = s | (1 << i)

start = time.time()
distance = TSP(s, init_point)
end = time.time()

for i in range(1, N):
    s = s | (1 << i)
init = 0
num = 0
print(distance)
route = [chosen_p]

while True:
    print(path[s][init])
    init = path[s][init]
    route.append(point_list[init])
    s = s & (~ (1 << init))
    num += 1
    if num > N - 2:
        break
print("程序的运行时间是：%s" % (end - start))

# 此代码为结果可视化展示
route.append(chosen_p)
plt.scatter(latitudes, longitude)
plt.scatter([chosen_p[0]], [chosen_p[1]], color='r')
x = [point[0] for point in route]
y = [point[1] for point in route]
plt.plot(x, y, color='black')
plt.show()