import matplotlib.pyplot as plt
import numpy as np
import os

def compare(ori, pre, sft, plot_shifted, title_list):  ## Use the new function
    for index in range(len(title_list)-1): #range(): #
        title = title_list[index]
        x = np.arange(1, len(ori)+1)
        ori_data = ori[:len(ori), index]
        pre_data = pre[:len(ori), index]
        sft_data = sft[:len(ori), index]
        plt.figure(figsize=(10,6))
        fig,ax = plt.subplots()
        ax.plot(x, ori_data, label='original data')
        ax.plot(x, pre_data, label='predict data')
        if plot_shifted: 
            ax.plot(x, sft_data, label='shifted original data')

        #plt.axvline(x=prediction_point, color='r', linestyle='--')
        mse = ((pre_data-ori_data)**2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(pre_data-ori_data).mean()
        R_sq = np.corrcoef(ori_data,pre_data)[0,1]**2
        print(f"for signal {index}: mse: {mse}, rmse: {rmse}, mae: {mae}, r_square:{R_sq}")
        ax.set_title(title)
        ax.legend()
        ax.margins(0.05)
        father_path = "caches/prediction"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        plt.savefig(os.path.join(father_path,f"{index}.png"))
        plt.close()