import os
from cv2 import cvtColor
import torch
import cv2
import numpy as np
from torchvision import transforms

from PIL import Image

from FOD.FocusOnDepth import FocusOnDepth
from FOD.j_utils import create_dir

# from FOD.j_dataset import show


class Predictor(object):
    def __init__(self, config, path_model, device="cpu"):
        self.config = config
        self.type = self.config["General"]["type"]

        # self.device = torch.device(
        #     self.config["General"]["device"] if torch.cuda.is_available() else "cpu"
        # )
        print(f"Device={device}")
        self.device = f"cuda:{device}"
        print(f"device: {self.device}")
        resize = config["Dataset"]["transforms"]["resize"]
        self.model = FocusOnDepth(
            image_size=(3, resize, resize),
            emb_dim=config["General"]["emb_dim"],
            resample_dim=config["General"]["resample_dim"],
            read=config["General"]["read"],
            nclasses=len(config["Dataset"]["classes"]) + 1,
            hooks=config["General"]["hooks"],
            model_timm=config["General"]["model_timm"],
            type=self.type,
            patch_size=config["General"]["patch_size"],
        )
        if path_model is None:
            path_model = os.path.join(
                config["General"]["path_model"],
                "FocusOnDepth_{}.p".format(config["General"]["model_timm"]),
            )
        # self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in torch.load(path_model, map_location=self.device)[
                    "model_state_dict"
                ].items()
            }
        )
        # self.model.load_state_dict(
        #     torch.load(path_model, map_location=self.device)["model_state_dict"]
        # )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def do(self, pil_im):
        original_size = pil_im.size
        # print("orginal_size=", original_size)  # WxH
        tensor_im = self.transform_image(pil_im).unsqueeze(0)  # type: ignore
        # print("tensor_im.shape", tensor_im.shape)
        output_depth, output_segmentation = self.model(tensor_im)
        output_depth = 1 - output_depth

        output_segmentation = transforms.ToPILImage()(
            output_segmentation.squeeze(0).argmax(dim=0).float()
        ).resize(original_size, resample=Image.NEAREST)
        output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(
            original_size, resample=Image.BICUBIC
        )
        return tuple(map(np.array, (pil_im, output_segmentation, output_depth)))

    def do_using_fname(self, image_fname):
        pil_im = Image.open(image_fname).convert("RGB")
        return self.do(pil_im)

    def get_image_seg_depth(self, mat, seg, depth):
        # normalizedImg = np.zeros(depth.shape)
        # depth = cv2.normalize(depth, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        seg, depth = (cvtColor(mat, cv2.COLOR_GRAY2RGB) for mat in (seg, depth))
        return np.hstack((mat, seg, depth))

    def get_image_seg_depth_using_fname(self, image_fname):
        return self.get_image_seg_depth(*self.do_using_fname(image_fname))

        # ## TO DO: Apply AutoFocus

        # output_depth = np.array(output_depth)
        # output_segmentation = np.array(output_segmentation)

        # mask_person = (output_segmentation != 0)
        # depth_person = output_depth*mask_person
        # mean_depth_person = np.mean(depth_person[depth_person != 0])
        # std_depth_person = np.std(depth_person[depth_person != 0])

        # #print(mean_depth_person, std_depth_person)

        # mask_total = (depth_person >= mean_depth_person-2*std_depth_person)
        # mask_total = np.repeat(mask_total[:, :, np.newaxis], 3, axis=-1)
        # #region_to_blur = np.ones(np_im.shape)*(1-mask_total)

        # #region_not_to_blur = np.zeros(np_im.shape) + np_im*(mask_total)
        # #region_not_to_blur = np_im
        # blurred = cv2.blur(region_to_blur, (10, 10))

        # #final_image = blurred + region_not_to_blur
        # final_image = cv2.addWeighted(region_not_to_blur.astype(np.uint8), 0.5, blurred.astype(np.uint8), 0.5, 0)
        # final_image = Image.fromarray((final_image).astype(np.uint8))
        # final_image.save(os.path.join(self.output_dir, os.path.basename(images)))

    def save_seg_depth(
        self, image_name, output_segmentation, output_depth, output_dir=None
    ):

        # with torch.no_grad():

        if output_dir is None:
            # output_dir = f"{self.config["General"]["path_predicted_images"]}_{self.config["General"]["resample_dim"]:04}",
            output_dir = "%s_%04d" % (
                self.config["General"]["path_predicted_images"],
                self.config["General"]["resample_dim"],
            )
        create_dir(output_dir)
        image_name = os.path.basename(image_name)
        path_dir_segmentation = os.path.join(output_dir, "segmentations")
        create_dir(path_dir_segmentation)
        cv2.imsave(
            os.path.join(path_dir_segmentation, image_name),
            output_segmentation,
        )

        path_dir_depths = os.path.join(output_dir, "depths")
        create_dir(path_dir_depths)
        cv2.imsave(os.path.join(path_dir_depths, image_name), output_depth)

    def run(self, input_images_fname, output_dir=None):
        for image_fname in input_images_fname:
            mats = mat, output_segmentation, output_depth = self.do(image_fname)
            cv2.imshow("mat, output_segmentation, output_depth", np.hstack(mats))
            cv2.waitKey(0)
            # self.save_seg_depth(image_fname, output_segmentation, output_depth, output_dir)
