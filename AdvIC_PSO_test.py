import cv2
import random
import numpy as np
import os
from detect_single_img import detect_v3_inf
from functions import initiation, scratch_pso, clip


dir_inf = r'C:\Users\a\PycharmProjects\pythonProject\yolo_v3_train\yolo_v3\data\custom\val_people'
# dir_vis = 'dataset_attack/vis/'


omega, c1, r1, c2, r2 = 0.9, 1.6, 0.5, 1.4, 0.5
seed = 50
step = 10

R = 1

ASR = [0, 0, 0, 0, 0, 0, 0]
Query = [0, 0, 0, 0, 0, 0, 0]
count_all = [0, 0, 0, 0, 0, 0, 0]

for K in range(0, 1):
    file = os.listdir(dir_inf)

    tag = 0
    for pic in file:

        tag = tag + 1

        if tag < 12:
            continue

        img_inf_path = dir_inf + '/' + pic
        # img_vis_path = dir_vis + pic

        # img_inf_path = r'C:\Users\a\Desktop\LLVIP\LLVIP\infrared\test\190002.jpg'
        # img_vis_path = r'C:\Users\a\Desktop\LLVIP\LLVIP\visible\test\190002.jpg'
        # path_adv_vis = 'adv_vis.jpg'
        path_adv_inf = 'adv.jpg'

        res_clean_inf = detect_v3_inf(img_inf_path)


        shape_inf, b1 = res_clean_inf.shape


        print('K, pic_id, shape_inf = ', K, tag, shape_inf)
        print('ASR = ', ASR)
        print('Query = ', Query)
        print('count_all = ', count_all)

        if shape_inf != 1:
            continue

        count_all[K] = count_all[K] + 1

        print('count_all = ', count_all)

        population = np.zeros((seed, (K+1)*6))
        unit = np.zeros((1, (K+1)*6))
        print('population = ', population)

        res = detect_v3_inf(img_inf_path)
        # print(res)
        # print(res[2])
        X1, Y1, X2, Y2 = int(res[0][0]), int(res[0][1]), int(res[0][2]), int(res[0][3])

        print(X1, Y1, X2, Y2)

        population = initiation(population, X1, Y1, X2, Y2)

        print('population = ', population)

        conf = np.zeros((1, seed))
        P_best = np.zeros((seed, (K+1)*6))
        conf_p = np.ones((1, seed)) * 100
        G_best = np.zeros((1, (K+1)*6))
        conf_G = 100
        V = np.zeros((seed, (K+1)*6))

        tag_break = 0
        # tag_run_vis = 0
        for steps in range(step):

            if tag_break == 1:
                break
            for seeds in range(seed):

                Query[K] = Query[K] + 1

                print('K, pic_id, steps, seeds = ', K, tag, steps, seeds)
                print('ASR = ', ASR)
                print('Query = ', Query)
                print('count_all = ', count_all)

                unit = population[seeds]
                # print('unit = ', unit)

                # img_inf = cv2.imread(img_inf_path)

                scratch_pso(img_inf_path, population[seeds], path_adv_inf, K+1, R)

                # img_show = plt.imread(path_adv_inf)
                # plt.imshow(img_show)
                # plt.show()

                res_inf = detect_v3_inf(path_adv_inf)
                print('res_inf.shape = ', res_inf.shape)

                if res_inf.shape == (0, 6):
                    G_best[0] = population[seeds]
                    tag_break = 1

                    img_inf_best = cv2.imread(path_adv_inf)
                    path_inf_best = 'path_adv/digital_inf1/' + pic
                    cv2.imwrite(path_inf_best, img_inf_best)

                    # tag_run_vis = 1

                    ASR[K] = ASR[K] + 1

                    break

                conf[0][seeds] = res_inf[0][4]

                # print('conf = ', conf)

                if conf[0][seeds] < conf_p[0][seeds]:  # 更新P_best
                    P_best[seeds] = population[seeds]
                    conf_p[0][seeds] = conf[0][seeds]

                # print(P_best)

                if conf[0][seeds] < conf_G:  # 更新G_best
                    G_best[0] = population[seeds]
                    conf_G = conf[0][seeds]

                    img_inf_best = cv2.imread(path_adv_inf)
                    path_inf_best = 'adv_inf_best.jpg'

                    cv2.imwrite(path_inf_best, img_inf_best)

                # print(G_best)

            for seeds in range(0, seed):
                for i in range(0, (K+1)*6):
                    V[seeds][i] = omega * V[seeds][i] + c1 * r1 * (
                            P_best[seeds][i] - population[seeds][i]) + c2 * r2 * (
                                          G_best[0][i] - population[seeds][i])
                    population[seeds][i] = population[seeds][i] + int(V[seeds][i])

            # print('V = ', V)

            # print('population = ', population)

            population = clip(population, X1, Y1, X2, Y2)










        # if count_all[0] == 1:
        #     break

print('ASR = ', ASR)
print('Query = ', Query)
print('count_all = ', count_all)

print(ASR[0]/count_all[0])
print(Query[0]/count_all[0])





