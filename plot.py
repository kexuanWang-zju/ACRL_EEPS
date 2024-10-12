import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np

def main():
    alpha_deg_reward = 100
    alpha_deg_cost = 100
    update_time_per_episode = 10
    episode = 210
    interval = 1
    x = np.zeros((episode,), dtype=np.int32)
    for jj in range(int(episode / interval)):
        x[jj] = jj
    constr_limit = 0.1*np.ones((episode,), dtype=np.int32)

    ############################################### encoder1 reshape1
    encoder_flag = 1
    shape_flag = 1
    Lv = 6
    name1 = "SLDAC_Lv" + str(Lv) + "_encoder" + str(encoder_flag) + "_shape" + str(shape_flag)
    SLDAC6_encoder1_shape1_reward = loadmat(name1+"_reward.mat")["array"]
    SLDAC6_encoder1_shape1_reward = np.mean(SLDAC6_encoder1_shape1_reward, axis=0)
    SLDAC6_encoder1_shape1_reward = SLDAC6_encoder1_shape1_reward[0:episode][::interval]
    SLDAC6_encoder1_shape1_cost = loadmat(name1+"_cost.mat")["array"]
    SLDAC6_encoder1_shape1_cost = np.mean(SLDAC6_encoder1_shape1_cost, axis=0)
    SLDAC6_encoder1_shape1_cost = SLDAC6_encoder1_shape1_cost[0:episode][::interval]

    Lv = 3
    name1 = "SLDAC_Lv" + str(Lv) + "_encoder" + str(encoder_flag) + "_shape" + str(shape_flag)
    SLDAC3_encoder1_shape1_reward = loadmat(name1+"_reward.mat")["array"]
    SLDAC3_encoder1_shape1_reward = np.mean(SLDAC3_encoder1_shape1_reward, axis=0)
    SLDAC3_encoder1_shape1_reward = SLDAC3_encoder1_shape1_reward[0:episode][::interval]
    SLDAC3_encoder1_shape1_reward = np.concatenate((SLDAC6_encoder1_shape1_reward, SLDAC3_encoder1_shape1_reward))
    #nihe1 = np.polyfit(x, SLDAC3_encoder1_shape1_reward, deg=alpha_deg_reward)
    #SLDAC3_encoder1_shape1_reward = np.polyval(nihe1, x)
    SLDAC3_encoder1_shape1_cost = loadmat(name1+"_cost.mat")["array"]
    SLDAC3_encoder1_shape1_cost = np.mean(SLDAC3_encoder1_shape1_cost, axis=0)
    SLDAC3_encoder1_shape1_cost = SLDAC3_encoder1_shape1_cost[0:episode][::interval]
    SLDAC3_encoder1_shape1_cost = np.concatenate((SLDAC6_encoder1_shape1_cost, SLDAC3_encoder1_shape1_cost))
    #nihe1 = np.polyfit(x, SLDAC3_encoder1_shape1_cost, deg=alpha_deg_cost)
    #SLDAC3_encoder1_shape1_cost = np.polyval(nihe1, x)

    Lv = 2
    name1 = "SLDAC_Lv" + str(Lv) + "_encoder" + str(encoder_flag) + "_shape" + str(shape_flag)
    SLDAC2_encoder1_shape1_reward = loadmat(name1+"_reward.mat")["array"]
    SLDAC2_encoder1_shape1_reward = np.mean(SLDAC2_encoder1_shape1_reward, axis=0)
    SLDAC2_encoder1_shape1_reward = SLDAC2_encoder1_shape1_reward[0:episode][::interval]
    SLDAC2_encoder1_shape1_reward = np.concatenate((SLDAC6_encoder1_shape1_reward, SLDAC2_encoder1_shape1_reward))
    #nihe1 = np.polyfit(x, SLDAC2_encoder1_shape1_reward, deg=alpha_deg_reward)
    #SLDAC2_encoder1_shape1_reward = np.polyval(nihe1, x)
    SLDAC2_encoder1_shape1_cost = loadmat(name1+"_cost.mat")["array"]
    SLDAC2_encoder1_shape1_cost = np.mean(SLDAC2_encoder1_shape1_cost, axis=0)
    SLDAC2_encoder1_shape1_cost = SLDAC2_encoder1_shape1_cost[0:episode][::interval]
    SLDAC2_encoder1_shape1_cost = np.concatenate((SLDAC6_encoder1_shape1_cost, SLDAC2_encoder1_shape1_cost))
    #nihe1 = np.polyfit(x, SLDAC2_encoder1_shape1_cost, deg=alpha_deg_cost)
    #SLDAC2_encoder1_shape1_cost = np.polyval(nihe1, x)

    Lv = 1
    name1 = "SLDAC_Lv" + str(Lv) + "_encoder" + str(encoder_flag) + "_shape" + str(shape_flag)
    SLDAC1_encoder1_shape1_reward = loadmat(name1+"_reward.mat")["array"]
    SLDAC1_encoder1_shape1_reward = np.mean(SLDAC1_encoder1_shape1_reward, axis=0)
    SLDAC1_encoder1_shape1_reward = SLDAC1_encoder1_shape1_reward[0:episode][::interval]
    SLDAC1_encoder1_shape1_reward = np.concatenate((SLDAC6_encoder1_shape1_reward, SLDAC1_encoder1_shape1_reward))
    #nihe1 = np.polyfit(x, SLDAC1_encoder1_shape1_reward, deg=alpha_deg_reward)
    #SLDAC1_encoder1_shape1_reward = np.polyval(nihe1, x)
    SLDAC1_encoder1_shape1_cost = loadmat(name1+"_cost.mat")["array"]
    SLDAC1_encoder1_shape1_cost = np.mean(SLDAC1_encoder1_shape1_cost, axis=0)
    SLDAC1_encoder1_shape1_cost = SLDAC1_encoder1_shape1_cost[0:episode][::interval]
    SLDAC1_encoder1_shape1_cost = np.concatenate((SLDAC6_encoder1_shape1_cost, SLDAC1_encoder1_shape1_cost))
    #nihe1 = np.polyfit(x, SLDAC1_encoder1_shape1_cost, deg=alpha_deg_cost)
    #SLDAC1_encoder1_shape1_cost = np.polyval(nihe1, x)

    
    ############################################### encoder1 shape1
    plt.figure(figsize=(9, 6.5))
    plt.plot(x, SLDAC3_encoder1_shape1_reward, color='blue', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC_Lv3')
    plt.plot(x, SLDAC2_encoder1_shape1_reward, color='red', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC_Lv2')
    plt.plot(x, SLDAC1_encoder1_shape1_reward, color='green', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC_Lv1')
    plt.margins(x=0)
    plt.ylim(0, 3)
    plt.xlabel("iteration")
    my_x_ticks_1 = np.arange(0, int(episode/interval), 20)
    my_x_ticks_2 = np.arange(0, update_time_per_episode*episode, update_time_per_episode*interval*20)
    plt.xticks(my_x_ticks_1, my_x_ticks_2)
    plt.ylabel('Hard-delay constrained effective throughout')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig("URLLC_reward_encoder1_shape1.pdf")
    #plt.show()

    plt.figure(figsize=(9, 6.5))
    plt.plot(x, SLDAC3_encoder1_shape1_cost, color='blue', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC_Lv3')
    plt.plot(x, SLDAC2_encoder1_shape1_cost, color='red', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC_Lv2')
    plt.plot(x, SLDAC1_encoder1_shape1_cost, color='green', linewidth=3, linestyle='-', marker="s", markersize=0.5,label='SLDAC_Lv1')
    plt.plot(x, constr_limit, color='black', linewidth=2, linestyle='-', label='constraint limit')
    plt.margins(x=0)
    plt.ylim(0, 0.5)
    plt.xlabel("iteration")
    my_x_ticks_1 = np.arange(0, int(episode/interval), 20)
    my_x_ticks_2 = np.arange(0, update_time_per_episode*episode, update_time_per_episode*interval*20)
    plt.xticks(my_x_ticks_1, my_x_ticks_2)
    plt.ylabel('Dropout Rate')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig("URLLC_cost_encoder1_shape1.pdf")
    #plt.show()
    

main()