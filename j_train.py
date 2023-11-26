# %env NCCL_DEBUG=INFO
import json
import os
import numpy as np
import torch
import platform
from FOD.j_tools import Hconf, Wtss
from FOD.j_Trainer import Trainer

from FOD.j_dataset import GenDataset

b_parallel_jobs = True


#生成数据加载器
# def gen_dataloader_from_list_data(config, list_data, type_str, debug_max_samples):
#     autofocus_datasets = [
#         AutoFocusDataset(config, dataset_name, type_str, debug_max_samples)
#         for dataset_name in list_data
#     ]
#     b_dones = [d.b_done for d in autofocus_datasets]
#     data = ConcatDataset(autofocus_datasets)
#     dataloader = DataLoader(
#         data, batch_size=config["General"]["batch_size"], shuffle=True
#     )
#     return dataloader, b_dones

#准备
def do(
    config,
    wts,
    job_idx=None,
    #抽取样本100进行训练，在我们的实验中我们要使用全部的样本
    num_samples=100,
    device_ids=[],
    image_folder=None,
    b_nn_parallel=True,
):
    assert image_folder is not None, "error: image_folder"
    hconf = Hconf(device_ids, b_nn_parallel)
    #显示进行到哪个epochs
    #这个写法值得我们参考
    print(
        "\n==== Doing %s (Epochs=%d) ===="
        % (wts.model_name(), config["General"]["epochs"])
    )
    # np.random.seed(config["General"]["seed"])

    #显示结果
    #     ==== Doing model_A070_P300_F040_C250_R010_T600_S050_M020_I050_O050_M050_Z050 (Epochs=100) ====
    #list_data是数据集的位置，基于config
    list_data = config["Dataset"]["paths"]["list_datasets"]

    #循环
    gds = [GenDataset(config, dataset_name) for dataset_name in list_data]
    trainer = Trainer(hconf, config, wts, job_idx)
    b_dones_val = trainer.train(gds, num_samples, image_folder)
    return all(b_dones_val)

#训练
def do2(dev_count, config, wtss, job_idx, debug_max_samples, image_folder, b_nn_parallel):
    #job_idx是训练的ID？
    print(f"---- Job ID = {job_idx}----")
    #print(platform.node())  # 计算机网络名称  yanzis-MacBook-Pro.local
    #如果计算机网络名称=MDL-001  ？
    if platform.node() == "MDL-001":
        device_id = job_idx % (dev_count - 1) + 1
    else:
        device_id = job_idx % dev_count

    do(
        config,
        wtss[job_idx],
        job_idx,
        debug_max_samples,
        [device_id],
        image_folder,
        b_nn_parallel,
    )


def main(num_samples, debug_max_samples=0):

    # debug_max_samples = None
    # debug_max_samples = 10

    #数据集位置为input
    # image_fname = "input/two_cats.png"
    image_folder = "input"
    #断言assert，如果出错就停止在这，我们希望它存在
    assert os.path.exists(image_folder)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 用GPU？
    torch.cuda.device_count()
    dev_count = int(torch.cuda.device_count())
    print("使用的GPU是: ", dev_count)
    #训练模型调用了config里面的参数
    with open("config.json", "r") as f:
        #调用json文件夹f，传给config
        config = json.load(f)
        #seed种子，表示每次使用np生成的随机数组相同
    np.random.seed(config["General"]["seed"])  # seed once only!
    
    #self.names在j_tools中定义，属于class Wts()

    # self.names = [
    #         0"depth_datum",
    #         1"loss_seg_penality_factor",       # too small => segmentation wt decrease; too big ==> cannot converge smoothly
    #         2"loss_fine_threshold_factor",     # too small => no improvement in smoothness, depth accurarcy and similarities; too big => segmentatin wt decreases
    #         3"loss_coarse_threshod_factor",    # too small => segmentatin wt increases, cannot converge smoothly; too big => segmentatin wt decreases, cannot converge smoothly; should be around 2
    #         4"loss_ratio_out_factor",          # too big (>10) => cannot determine in vs out, too small(<2) => cannot converge
    #
    #         5"loss_ratio_out_attenuation_factor"   # too big => segment wt will outweight others when epoch increases, too small => veritical striation appears when epoch increases
    #         6"loss_segmentation_factor",                   # segmentation wt
    #         7"loss_mse_factor",                            # mse wt
    #         8"loss_depth_in_factor",                       # depth in wt
    #         9"loss_depth_out_factor",                      # depth out wt
    #         10"loss_smooth_factor",                        # edge wt  ;too small => smoothness decrease
    #         11"loss_ssim_factor",                          # similarities wt
    #     ]

    #Wtss是一个权重辅助类，这样可以更容易地定义所有这些值
    wtss = Wtss()
    #          0    1    2    3    4    5    6    7    8    9   10   11
    wtss.add(0.7, 3.0, 0.4, 2.5, 0.1, 6.0, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5)    # good
    wtss.add(0.7, 3.0, 0.4, 2.5, 0.1, 6.0, 0.5, 0.2, 0.6, 0.4, 0.5, 0.5)    # good

    #len用于获取长度  len(wtss=) 2
    print("len(wtss)=", len(wtss))

    #n_jobs进程数

    n_jobs = min(max(1, dev_count), len(wtss))
    if platform.node() == "MDL-001":
        n_jobs -= 1

    print("n_jobs=", n_jobs)
    # wts_list = list(wtss.values())[:n_jobs]
    #将wtss转化为列表（本来就是？）
    wts_list = list(wtss.values())



    if dev_count <= 0 or b_parallel_jobs is False:
        for job_idx, wts in enumerate(wts_list):
            if do(config, wts, job_idx, num_samples, [0], image_folder):
                break
    else:
        
        import joblib
    #joblib用于保存模型到磁盘  parallel方法并发进行训练 n_jobs进程数
        print("do2 joblib.........")
        b_nn_parallel = False if (n_jobs >= dev_count) else True
        assert image_folder is not None, "error: image_folder"
        result = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(do2)(
                dev_count,
                config,
                wts_list,
                job_idx,
                num_samples,
                image_folder,
                b_nn_parallel,
            )
            for job_idx in range(len(wts_list))
            # for job_idx in range(len(wts_list))
        )
        # with open("config.json", "r") as f:
        print(f"Result={result}")


if __name__ == "__main__":
    global b_display_ok
    b_display_ok = False
    num_samples_first = 20
    # 20 , 40   数字越大，精度越高，耗时越多
    # num_samples_first should not be bigger than 40(default is 20) for code developing
    # num_samples_first should not be smaller than 80(default is ALL) for generatring finalised model
    num_list_items = 1
    mlist = list(num_samples_first * 2 ** np.arange(num_list_items))  # (20, 40, 80, 160, 320, 640)
    for num_samples in mlist:
        main(num_samples)




