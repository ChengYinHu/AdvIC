import cv2
import random
import numpy as np
import os

def initiation(POP, X1, Y1, X2, Y2):

    a, b = POP.shape[0], POP.shape[1]
    for i in range(0, a):
        for j in range(0, b):
            if j%2 == 0:
                POP[i][j] = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3))

            if j%2 == 1:
                POP[i][j] = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 3))


    return POP


def scratch_random(img_path, X1, Y1, X2, Y2, path_adv, K, R):

    img = cv2.imread(img_path)
    for k in range(K):
        x1, x2, x3 = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)), random.randint(
            int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)), random.randint(int(X1 + (X2 - X1) // 3),
                                                                                int(X2 - (X2 - X1) // 3))
        y1, y2, y3 = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 4)), random.randint(
            int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 4)), random.randint(int(Y1 + (Y2 - Y1) // 5),
                                                                                int(Y2 - (Y2 - Y1) // 4))

        # print('x1, y1, x2, y2, x3, y3 = ', x1, y1, x2, y2, x3, y3)

        points = [
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ]

        points = np.array(points)  # 把列表转换为数组

        def calNextPoints(points, rate):  # 如果给定了具体的n， 那么可以直接得到计算方程
            if len(points) == 1:
                return points  # 若最后一个点，返回

            left = points[0]
            ans = []
            for i in range(1, len(points)):  # 根据比例计算当前的点的坐标，一层层的推进
                right = points[i]
                disX = right[0] - left[0]
                disY = right[1] - left[1]

                nowX = left[0] + disX * rate
                nowY = left[1] + disY * rate
                ans.append([nowX, nowY])  # 将nowX和nowY填入ans数组中

                left = right  # 更新left

            return calNextPoints(ans, rate)  # 继续递归

        X = []
        Y = []
        for i in range(0, 100):  # 循环次数越多，画出的图形越平滑
            i = i / 100
            a = calNextPoints(points, rate=i)  # 计算位置
            x = a[0][0]
            y = a[0][1]
            X.append(x)
            Y.append(y)

        # print('X = ', X)
        # print('Y = ', Y)
        # print(X[1])
        # print(X[98])
        # print(X[99])
        # print(X[100])

        for i in range(0, 100):

            cv2.circle(img, (int(X[i]), int(Y[i])), R, (0, 0, 0), -1)

    cv2.imwrite(path_adv, img)

def line(img_path, X1, Y1, X2, Y2, path_adv, K, R):

    img = cv2.imread(img_path)
    for k in range(K):
        x1, x2 = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)), random.randint(int(X1 + (X2 - X1) // 3),  int(X2 - (X2 - X1) // 3))
        y1, y2 = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 2)), random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 2))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), R)

    cv2.imwrite(path_adv, img)

def triangle(img_path, X1, Y1, X2, Y2, path_adv, R):
    img = cv2.imread(img_path)
    for k in range(1):
        x1, x2, x3 = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)), random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)), random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3))
        y1, y2, y3 = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 4)), random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 4)), random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 4))
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 0), R)

    cv2.imwrite(path_adv, img)

def circle(img_path, X1, Y1, X2, Y2, path_adv):

    img = cv2.imread(img_path)

    x, y = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)), random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 4))

    # x, y = int(X1+(X2-X1)//2), int(Y1+(Y2-Y1)//3)

    R = int((X2-X1)/3)

    # R = int((X2 - X1) / 4)

    cv2.circle(img, (x, y), R, (0, 0, 0), 2)

    cv2.imwrite(path_adv, img)







def scratch_pso(img_path, POP, path_adv, K, R):

    img = cv2.imread(img_path)
    for k in range(K):

        x1, y1 = POP[6 * k + 0], POP[6 * k + 1]
        x2, y2 = POP[6 * k + 2], POP[6 * k + 3]
        x3, y3 = POP[6 * k + 4], POP[6 * k + 5]


        # print('x1, y1, x2, y2, x3, y3 = ', x1, y1, x2, y2, x3, y3)

        points = [
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ]

        points = np.array(points)  # 把列表转换为数组

        def calNextPoints(points, rate):  # 如果给定了具体的n， 那么可以直接得到计算方程
            if len(points) == 1:
                return points  # 若最后一个点，返回

            left = points[0]
            ans = []
            for i in range(1, len(points)):  # 根据比例计算当前的点的坐标，一层层的推进
                right = points[i]
                disX = right[0] - left[0]
                disY = right[1] - left[1]

                nowX = left[0] + disX * rate
                nowY = left[1] + disY * rate
                ans.append([nowX, nowY])  # 将nowX和nowY填入ans数组中

                left = right  # 更新left

            return calNextPoints(ans, rate)  # 继续递归

        X = []
        Y = []
        for i in range(0, 100):  # 循环次数越多，画出的图形越平滑
            i = i / 100
            a = calNextPoints(points, rate=i)  # 计算位置
            x = a[0][0]
            y = a[0][1]
            X.append(x)
            Y.append(y)

        # print('X = ', X)
        # print('Y = ', Y)
        # print(X[1])
        # print(X[98])
        # print(X[99])
        # print(X[100])

        for i in range(0, 100):

            cv2.circle(img, (int(X[i]), int(Y[i])), R, (0, 0, 0), -1)

    cv2.imwrite(path_adv, img)


def clip(population, X1, Y1, X2, Y2):

    a, b = population.shape

    # print('a, b = ', a, b)

    for i in range(0, a):

        for j in range(0, b):

            if j%2 == 0:
                if population[i][j] not in range(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)):
                    population[i][j] = random.randint(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3))
            if j%2 == 1:
                if population[i][j] not in range(int(X1 + (X2 - X1) // 3), int(X2 - (X2 - X1) // 3)):
                    population[i][j] = random.randint(int(Y1 + (Y2 - Y1) // 5), int(Y2 - (Y2 - Y1) // 3))





    return population
