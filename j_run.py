import json
from glob import glob
from FOD.j_Predictor import Predictor
import os
import cv2
import numpy as np

#方法do
def do(outname=None, b_stack_order_col_then_row=True, b_intermediate_show=False):
    np_stacks = (np.hstack, np.vstack)
    if b_stack_order_col_then_row:
        np_stacks = np_stacks[::-1]

    with open("config.json", "r") as f:
        config = json.load(f)

    input_images = glob("input/*.jpg") + glob("input/*.png")
    # dst_dirs = ("output_1", "output_2")
    # model_dirs = ("models_1", "models_2")
    outs = []
    outss = []
    for src in input_images:
        # for (dst_dir, model_dir) in zip(dst_dirs, model_dirs):
        for i in range(1, 10):
            model_dir = "models_%d" % i

            if not os.path.exists(model_dir):
                print("%s not found" % model_dir)
                break
            #模型路径
            path_model = os.path.join(model_dir, "FocusOnDepth.p")
            if not os.path.exists(path_model):
                continue

            predictor = Predictor(config, path_model)
            dst = predictor.get_image_seg_depth_using_fname(src)
            if b_intermediate_show:
                cv2.imshow("dst", dst)
                cv2.waitKey()
            outs.append(dst)
        outss.append(np_stacks[0](outs))

    out = np_stacks[1](outss) if len(outss) > 1 else outss[0]
    cv2.imshow("out", out)
    cv2.waitKey()
    if outname:
        print(f"writing {outname}")
        cv2.imwrite(outname, out)


if __name__ == "__main__":
    do("out.jpg")
