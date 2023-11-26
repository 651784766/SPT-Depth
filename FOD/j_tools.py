import torch
import torch.nn as nn
import nvsmi
import time
import numpy as np
import os
import itertools
import glob


class Hconf:
    def __init__(self, device_ids=[], b_nn_parallel=True, b_data_parallel=False):
        self.b_nn_parallel = b_nn_parallel
        self.b_data_parallel = b_data_parallel
        self.dev_count = torch.cuda.device_count()
        self.b_use_all_gpus = True
        self.device_ids_init(device_ids)

    def device_ids_init(self, device_ids):
        # gpus = list(nvsmi.get_available_gpus())
        gpus = list(nvsmi.get_gpus())
        num_available_gpus = len(gpus)
        print("num_available_gpus=", num_available_gpus)
        num_device_ids = len(device_ids)
        if num_available_gpus == 0 or self.dev_count == 0:
            self.device_ids = None
            self.device = "cpu"
            torch.device(self.device)
        else:
            available_gpu_ids = [int(gpu.id) for gpu in gpus]
            print("available_gpu_ids=", available_gpu_ids)
            print("device_ids=", device_ids)
            if num_device_ids > 0 and set(device_ids).issubset(set(available_gpu_ids)):
                self.device_ids = device_ids
            else:
                self.device_ids = (
                    available_gpu_ids if self.b_use_all_gpus else [available_gpu_ids[0]]
                )
            self.device = self.device_ids[0]
            torch.cuda.set_device("cuda:%s" % self.device)

        print("self.device = ", self.device)
        print("self.device_ids = ", self.device_ids)


class Wtss(dict):
    def add(self, *args, **kwargs):
        wts = Wts(*args, **kwargs)
        # print("wts.gen_name()=", wts.gen_name())
        self[wts.name()] = wts

    def info(self):
        print("===== Wtss ====")
        for key, val in self.items():
            # print("key=%s" % key)
            val.info()

    def list(self):
        return list(self.values())


class Wts(dict):
    def __init__(self, *args, **kwargs):
        # values of all wts is be between -1 and 1
        self.names = [
            "depth_datum",
            "loss_seg_penality_factor",
            "loss_fine_threshold_factor",
            "loss_coarse_threshold_factor",
            "loss_ratio_out_factor",
            "loss_ratio_out_attenuation_factor",
            "loss_segmentation_factor",
            "loss_mse_factor",
            "loss_depth_in_factor",
            "loss_depth_out_factor",
            "loss_smoothness_factor",
            "loss_ssim_factor",
        ]
        self.heads = "APFCRTSMIOMZ"
        if len(args) == len(self.names):
            dict = {key: wt for key, wt in zip(self.names, args)}
        elif len(kwargs) == len(self.names):
            dict = kwargs
        else:
            print("len(args)=", len(args))
            print("len(self.names)=", len(self.names))
            assert False, "[Params: __init__]fatal error!"
        # print("\ndict=", dict)
        self.update(dict)
        self.__dict__.update(dict)

    def name(self):
        mstr = "%03d_".join(self.heads) + "%03d"
        vals = tuple(np.array(tuple((self.values()))) * 100)
        return mstr % vals

    def model_name(self):
        return f"model_{self.name()}"

    def info(self):
        print("\n=== Wts[%s]===" % self.name())
        print("info=", self.__dict__)
        print("values=", self.values())
        print("keys=", self.keys())
        print("items=", self.items())
        print("\n")
        # print("name=", self.gen_name())


def getMaxGPUsTemperature():
    temperatures = []
    try:
        for gpu in nvsmi.get_gpus():
            temperatures.append(gpu.temperature)
        max_temperature = max(temperatures)
        # print("max_temp=", max_temperature)
        return max_temperature
    except Exception:
        return 0


def sleep_to_cool_down(threshold_temperature=80):
    sleep_seconds = 5
    total_sleep_seconds = 0
    while True:
        current_temperature = getMaxGPUsTemperature()
        print("current temperature = ", current_temperature)
        if current_temperature < threshold_temperature:
            break
        time.sleep(sleep_seconds)
        total_sleep_seconds += sleep_seconds
        print("total sleep time = %ds" % total_sleep_seconds)


def data_parallel(
    module, input, hconf
):  # should I restrict to use a specific gpu instead of multiple gpus ??
    # if len(hconf.device_ids) is None:
    #     return module(input)

    if hconf.dev_count <= 1 or not hconf.b_data_parallel:
        return module(input)

    output_device = hconf.device_ids[0] if hconf.device is None else hconf.device

    replicas = nn.parallel.replicate(module, hconf.device_ids)
    inputs = nn.parallel.scatter(input, hconf.device_ids)
    replicas = replicas[: len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


def get_images_from_folder(image_folder, exts=("jpg", "png", "jpeg")):
    return list(
        itertools.chain(
            *[glob.glob(os.path.join(image_folder, f"*.{ext}")) for ext in exts]
        )
    )


def get_images(*args):
    ret = []
    for arg in args:
        if os.path.isdir(arg):
            ret += get_images_from_folder(arg)
        elif os.path.exists(arg):
            ret.append(arg)
    return ret


if __name__ == "__main__":
    # Hconf().device_ids_init([])
    # p0 = Wts(1, 2, -3, 4)
    # p0.info()
    # p1 = Wts(depth_datum=1, segmentation=2, depth_in=3, depth_out=4)
    # p1.info()
    a = Wtss()
    a.add(1, 2, 3, 4)
    a.add(depth_datum=1, segmentation=2, depth=0.1, depth_in=0.3, depth_out=0.4)
    a.info()
    b = a.list()
    print(b[0].model_name())
    print(b[0]["depth_in"])
    b[0].a = 10
    b[0].__dict__["b"] = 10
    print(dir(b[0]))
