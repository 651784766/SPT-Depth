# pyright: reportUnboundVariable=false

import os
import torch
import numpy as np
import wandb
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

from pytorch_msssim import ssim  # , SSIM, MS_SSIM

# from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# from os import replace

from numpy.core.numeric import Inf  # type: ignore
from FOD.j_utils import get_losses, get_optimizer, get_schedulers, create_dir
from FOD.j_FocusOnDepth import FocusOnDepth
from FOD.j_tools import sleep_to_cool_down, get_images
from FOD.j_Predictor import Predictor
from FOD.j_Loss import ScaleAndShiftInvariantLoss, SmoothnessLoss
from FOD.j_dataset import gen_dataloader_from_list_data
b_epoch_0_pure_segementation = True
b_display_imgs = False
b_display_plots = False
# b_valid_use_loss_seg_penality = True
# b_valid_use_loss_smoothness = True
# b_valid_use_loss_ssim = True
# b_valid_use_ratio_out = True
# b_valid_use_mse = True
b_valid_use_loss_seg_penality, b_valid_use_loss_smoothness, b_valid_use_loss_ssim, b_valid_use_ratio_out, b_valid_use_mse = [True] * 5
b_valid_use_ratio_out = False


def get_ratio_out_raw(segmentations, output_depths):
    # print("type(segmentations)=", type(segmentations), segmentations)
    # segmentaions_mean = torch.mean(segmentations) if type(segmentations) in ("torchj.Tensor",) else 0.1
    segmentations_out = segmentations > 0
    # segmentations_in = torch.logical_not(segmentations_out)
    output_depths_out = output_depths > 0.7            # torch.mean(output_depths)
    # output_depths_in = torch.logical_not(output_depths_out)
    segmentations_out_count_non_zero = torch.count_nonzero(segmentations_out)
    # segmentations_in_count_non_zero = torch.count_nonzero(segmentations_in)

    ratio_out = torch.count_nonzero(torch.logical_and(output_depths_out, segmentations_out)) / (segmentations_out_count_non_zero + 1)
    # ratio_in = torch.count_nonzero(torch.logical_and(output_depths_in, segmentations_in)) / (segmentations_in_count_non_zero + 1)
    # print(".......................ratio_out=", ratio_out)
    # print("......................ratio_in=", ratio_in)
    # print("segmentations_in_count_non_zero=", segmentations_in_count_non_zero)
    # print("segmentations_out_count_non_zero=", segmentations_out_count_non_zero)
    # return (ratio_out + ratio_in) / 2.
    # return torch.min(ratio_out, ratio_in)
    return ratio_out


#用于画图，绘画各种损失函数的图像

def j_plots(figname, losses_val, losses_val_fine, losses, losses_seg):
    # print(f"image_folder={image_folder}")
    # print(f"self.device={self.device}")
    print("losses=", np.round(losses, 2))
    print("losses_seg=", np.round(losses_seg, 2))
    print("losses_val=", np.round(losses_val, 2))
    plt.figure()
    plt.plot(losses_val, label="losses_val")
    plt.plot(losses_val_fine, label="losses_val_fine")
    plt.plot(losses, label="losses")
    plt.plot(losses_seg, label="losses_seg")
    plt.legend()
    plt.savefig(figname)
    plt.show()


def show_image(mat):
    global b_display_imgs
    if b_display_imgs:
        try:
            # cv2.imshow(out_name, mat)
            # cv2.waitKey(500)                  # will crash if ssh without -X
            x = Image.fromarray(mat)
            # x = Image.open(out_name).convert('RGB')
            if x:
                ret = x.show()
                if not ret:
                    raise ValueError("cannot do pil.show()")
        except Exception:
            print("cannot show image!!! Try connect ssh with -X")
            print("eg. ssh -X joe@202.130.124.137")
            # global b_display_imgs
            b_display_imgs = False


def gen_image_basename(config, model_name, num_of_samples, epoch):
    loss_depth_method = config["General"]["loss_depth"]
    str_shuffled = "shuffled" if config["General"]["b_shuffle"] else ""
    return f"{model_name}_{num_of_samples}_ep{epoch:03d}-{loss_depth_method}-{str_shuffled}"


def show_images(
    image_folder,
    config,
    out_dir,
    model_fname,
    num_of_samples,
    val_loss,
    loss,
    device,
    basename
):

    images = get_images(image_folder)
    # print(f"images={images}")
    print(f"device@show_images={device}")
    predictor = Predictor(config, model_fname, device)
    out_dir_old = os.path.join(out_dir, f"{num_of_samples}")
    out_dir = os.path.join(out_dir, f"{num_of_samples:05}")
    out_up_one = os.path.dirname(out_dir)
    if not os.path.isdir(out_up_one):
        os.makedirs(out_up_one)
    if not os.path.isdir(out_dir):
        if os.path.isdir(out_dir_old):
            try:
                os.rename(out_dir_old, out_dir)
            except Exception:
                print(f"can't rename {out_dir_old} to {out_dir}")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    str_loss = "vs".join([f"{int(min(loss, 99)*1000):04d}" for loss in (val_loss, loss)])
    for image_fname in images:
        head, tail = os.path.splitext(os.path.basename(image_fname))
        image_fname = os.path.realpath(image_fname)
        # print("image_fname=", image_fname)
        mat = predictor.get_image_seg_depth_using_fname(image_fname)
        out_name = os.path.realpath(
            os.path.join(
                out_dir,
                f"{basename}_{str_loss}-{head}.jpg"
                # f"{model_name}_{num_of_samples}_ep{epoch:03d}_{str_loss}-{head}-{loss_depth_method}-{str_shuffled}.jpg",
            )
        )
        # print("out_name=", out_name)
        cv_mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_name, cv_mat)
        show_image(mat)
        print("Done Training")


class Trainer(object):
    def __init__(self, hconf, config, wts, job_idx=None, output_dir=None):
        super().__init__()
        self.config = config
        if output_dir is None:
            # output_dir = f"{self.config["General"]["path_predicted_images"]}_{self.config["General"]["resample_dim"]:04}",
            output_dir = "%s_%04d" % (
                self.config["General"]["path_predicted_images"],
                self.config["General"]["resample_dim"],
            )
        self.job_idx = job_idx
        self.config = config
        self.p = wts
        self.type = self.config["General"]["type"]
        self.output_dir = output_dir
        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")

        #
        #self.device = hconf.device
        print("device: %s" % self.device)
        resize = config["Dataset"]["transforms"]["resize"]
        self.model = FocusOnDepth(
            hconf,
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

        self.model.to(self.device)
        # print(self.model)
        # exit(0)
        self.loss_depths = [ScaleAndShiftInvariantLoss(), nn.MSELoss()]
        self.loss_depths_idx = 0
        self.loss_failed_count = 0
        self.loss_seg_min = None
        self.loss_depth, self.loss_segmentation = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(
            config, self.model
        )
        self.schedulers = get_schedulers(
            [self.optimizer_backbone, self.optimizer_scratch]
        )
        # Specify a path
        self.PATH = "state_dict_model.pt"

        # Save
        torch.save(self.model.state_dict(), self.PATH)

#调参，减少loss_failed_count_threshold到3可能是个好的选择

    def train(
        self, gds, num_samples, image_folder=None, loss_failed_count_threshold=5
    ):
        # train set
        train_dataloader, b_dones_train = gen_dataloader_from_list_data(
            gds, "train", num_samples
        )
        # validation set
        val_dataloader, b_dones_val = gen_dataloader_from_list_data(
            gds, "val", num_samples
        )
        epochs = self.config["General"]["epochs"]
        if self.config["wandb"]["enable"]:
            wandb.init(project="FocusOnDepth", entity=self.config["wandb"]["username"])
            wandb.config = {
                "learning_rate_backbone": self.config["General"]["lr_backbone"],
                "learning_rate_scratch": self.config["General"]["lr_scratch"],
                "epochs": epochs,
                "batch_size": self.config["General"]["batch_size"],
            }
        val_loss = Inf
        loss = Inf
        losses = []
        losses_seg = []
        losses_val = []
        losses_val_fine = []
        losses_ratio = []
        # b_resetted = False
        epoch = -1

        #***对数据集进行循环
        for i in range(epochs):  # loop over the dataset multiple times
            epoch += 1
            print("Epoch = ", epoch)
            sleep_to_cool_down()
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            description = f"<ep#{epoch}/{epochs}>Training"
            if self.job_idx is not None:
                description = f"[{self.job_idx}]{description}"
            losses_ep = []
            losses_seg_ep = []
            losses_ratio_ep = []
            pbar.set_description(description)
            for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
                if i % 100 == 0:
                    sleep_to_cool_down()
                # get the inputs; data is a list of [inputs, labels]
                X, Y_depths, Y_segmentations = (
                    X.to(self.device),
                    Y_depths.to(self.device),
                    Y_segmentations.to(self.device),
                )
                # zero the parameter gradients
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer
                output_depths, output_segmentations = self.model(X)
                output_depths = torch.nan_to_num(output_depths)
                output_segmentations = torch.nan_to_num(output_segmentations)

                output_depths = (
                    output_depths.squeeze(1) if output_depths is not None else None
                )  # must!

                Y_depths = Y_depths.squeeze(1)  # 1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1)  # 1xHxW -> HxW
                # get loss
                assert epoch >= 0
                loss_new, loss_seg_new, loss_ratio_out = self.j_loss(
                    output_depths,
                    Y_depths,
                    output_segmentations,
                    Y_segmentations,
                    epoch,
                )
                # assert torch.isnan(loss_new).sum() == 0, print(loss_new)
                # self.optimizer_scratch.zero_grad()
                # self.optimizer_backbone.zero_grad()
                loss_new.backward()

                # clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                # step optimizer
                # assert torch.isnan(self.model.mu).sum() == 0, print(self.model.mu)
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()
                # assert torch.isnan(self.model.mu).sum() == 0, print(self.model.mu)
                # assert torch.isnan(self.model.mu.grad).sum() == 0, print(self.model.mu.grad)
                # if loss < loss_new:
                #     continue
                if self.loss_seg_min is None or self.loss_seg_min > loss_seg_new:
                    self.loss_seg_min = loss_seg_new
                loss_float = loss_new.item()
                # print("loss_float=", loss_float)
                losses_ep.append(loss_float)
                losses_seg_ep.append(loss_seg_new.item())
                losses_ratio_ep.append(loss_ratio_out.item())
                running_loss += loss_float
        #从这里开始（被写进了for循环中）
                if np.isnan(running_loss):
                    print(
                        "\n",
                        X.min().item(),
                        X.max().item(),
                        "\n",
                        Y_depths.min().item(),
                        Y_depths.max().item(),
                        "\n",
                        output_depths.min().item()
                        if output_depths is not None
                        else None,
                        output_depths.max().item()
                        if output_depths is not None
                        else None,
                        "\n",
                        loss_float,
                    )
                    exit(0)

                if self.config["wandb"]["enable"] and (
                    (i % 50 == 0 and i > 0) or i == len(train_dataloader) - 1
                ):
                    wandb.log({"loss": running_loss / (i + 1)})
                pbar.set_postfix({"training_loss": running_loss / (i + 1)})
            assert epoch >= 0

            loss = np.mean(losses_ep)
            losses.append(loss)
            losses_seg.append(np.mean(losses_seg_ep))

            losses_ratio_ep_mean = np.mean(losses_ratio_ep)
            losses_ratio.append(losses_ratio_ep_mean)
            print("------------------------------------------------------losses_ratio_ep_mean=", losses_ratio_ep_mean)
            print(f"Y_depths.std()={Y_depths.std()}")

            new_val_loss, new_val_loss_fine = self.run_eval(val_dataloader, epoch)
            # print("new_loss_ratio=", new_loss_ratio)

            # # if new_loss_ratio > 0.05:
            # # if losses_ratio_ep_mean > 0.5:
            # if losses_ratio_ep_mean > 1.75:
            #     print("x" * 200)
            #     self.model.load_state_dict(torch.load(self.PATH))
            #     val_loss = Inf
            #     epoch = 0
            #     b_resetted = True
            #     continue
            losses_val.append(new_val_loss)
            losses_val_fine.append(new_val_loss_fine)
            print("\n" + f"new_val_loss = {new_val_loss:0.4f} vs {val_loss:0.4f}")
        #从Trainer.py中，您可以看到程序比较了new_val_Loss和旧的_val_loss
        #如果新的epoch的模型损失降低，将模型保存
        #     if new_val_loss < val_loss:
         #    self.save_model()    


            if new_val_loss < val_loss:   # and losses_ratio_ep_mean < 0.5:
                if epoch > 0:
                    val_loss = new_val_loss
                    self.loss_failed_count = 0
                    model_fname = self.save_model()
                    basename = gen_image_basename(self.config, self.p.model_name(), pbar.total, epoch)
                    if b_display_plots:
                        j_plots(f"{basename}_plt.jpg", losses_val, losses_val_fine, losses, losses_seg)

                    if image_folder:
                        print("=" * 200)
                        show_images(
                            image_folder,
                            self.config,
                            self.output_dir,
                            model_fname,
                            pbar.total,
                            val_loss,
                            loss,
                            self.device,
                            basename
                        )
            else:
                self.loss_failed_count += 1
                print(f"losses_ratio_ep_mean={losses_ratio_ep_mean}")
                print(
                    f"\n[{self.job_idx}]{epoch:03d} is skipped because the validation loss is not converging!!!! or losses_ratio_ep_mean too small"
                )
                print(
                    f"New val_loss = {new_val_loss:.4f} vs Previous val_loss = {val_loss:.4f}\n"
                )
            print(
                f"[{self.job_idx}]<ep#{epoch}/{epochs}>loss failed count = {self.loss_failed_count}"
            )
            print(
                f"[{self.job_idx}]<ep#{epoch}/{epochs}>loss_depths_idx = {self.loss_depths_idx}"
            )
            self.schedulers[0].step(new_val_loss)
            self.schedulers[1].step(new_val_loss)
            if self.loss_failed_count >= loss_failed_count_threshold:
                break
        # end for
        print("Finished Training")
        return b_dones_val
  

    
    def run_eval(self, val_dataloader, epoch):
        """
        ******************************
        Evaluate the model on the validation set and visualize some results
        on wandb
        在验证集上评估模型，并将部分结果在wandb上可视化
        
        :- val_dataloader -: torch dataloader
        """
        val_loss = 0.0
        self.model.eval()
        X_1 = None
        Y_depths_1 = None
        Y_segmentations_1 = None
        output_depths_1 = None
        output_segmentations_1 = None

        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
                X, Y_depths, Y_segmentations = (
                    X.to(self.device),
                    Y_depths.to(self.device),
                    Y_segmentations.to(self.device),
                )
                output_depths, output_segmentations = self.model(X)
                output_depths = torch.nan_to_num(output_depths)
                output_segmentations = torch.nan_to_num(output_segmentations)
                # output_depths = (
                #     output_depths.squeeze(1) if output_depths is not None else None
                # )
                output_depths = output_depths.squeeze(1)

                Y_depths = Y_depths.squeeze(1)
                Y_segmentations = Y_segmentations.squeeze(1)
                if i == 0:
                    X_1 = X
                    Y_depths_1 = Y_depths
                    Y_segmentations_1 = Y_segmentations
                    output_depths_1 = output_depths
                    output_segmentations_1 = output_segmentations
                assert epoch >= 0
                loss, loss_seg_new, loss_ratio_out = self.j_loss(
                    output_depths, Y_depths, output_segmentations, Y_segmentations, epoch
                )

                loss_fines = []
                if b_valid_use_loss_seg_penality:
                    if self.loss_seg_min is not None and self.loss_seg_min < loss_seg_new:
                        loss_seg_penality = (
                            loss_seg_new - self.loss_seg_min
                        ) * self.p.loss_seg_penality_factor
                        print("loss_seg_penality=", loss_seg_penality.item())
                        loss_fines.append(loss_seg_penality)

                if b_valid_use_loss_smoothness:
                    loss_smoothness = self.p.loss_smoothness_factor * SmoothnessLoss()(
                        Y_depths, output_depths
                    )
                    print("loss_smoothness=", loss_smoothness.item())
                    loss_fines.append(loss_smoothness)

                if b_valid_use_loss_ssim:
                    loss_ssim = self.p.loss_ssim_factor * torch.mean(
                        1 - ssim(Y_depths[None, :], output_depths[None, :])
                    )
                    print("loss_ssim=", loss_ssim.item())
                    loss_fines.append(loss_ssim)

                if b_valid_use_ratio_out:
                    # loss_ratio = ((1 - ratio_out) + (1 - ratio_in)) * self.p.loss_ratio_out_factor
                    ratio_out = get_ratio_out_raw(Y_segmentations, output_depths)
                    loss_ratio = (1 - ratio_out) * self.p.loss_ratio_out_factor
                    print("ratio_out=", ratio_out.item())
                    print("loss_ratio=", loss_ratio.item())
                    loss_fines.append(loss_ratio)

                if b_valid_use_mse:
                    # loss_mse_threshold = 0.005
                    # loss_mse_threshold_factor = 1.1
                    loss_mse = self.p.loss_mse_factor * nn.MSELoss()(output_depths, Y_depths)
                    print("loss_mse=", loss_mse.item())
                    loss_fines.append(loss_mse)

                loss_fine_factor = 1.       # use it to adjust the wt of fine loss vs coarse loss
                if len(loss_fines) > 0:
                    loss_fine = sum(loss_fines) * loss_fine_factor
                    print("loss_fine=", loss_fine)
                    # loss_fine = max(loss_smoothness, loss_ssim, loss_mse)
                    loss += loss_fine       # is it a must?
                # 精细收敛因子=0.5
                # loss_fine_threshold_factor = 0.5
                # loss = loss_fine
                # fx = 1.
                # loss = loss * (1 + loss_fine / loss2 * self.p.loss_fine_threshold_factor * fx )
                # loss = (
                #     loss2
                #     if loss2 < loss * loss_fine_threshold_factor
                #     else loss * loss_fine_threshold_factor
                # )
                val_loss += loss.item()
                pbar.set_postfix({"validation_loss": val_loss / (i + 1)})
            if self.config["wandb"]["enable"]:
                wandb.log({"val_loss": val_loss / (i + 1)})
                self.img_logger(
                    X_1,
                    Y_depths_1,
                    Y_segmentations_1,
                    output_depths_1,
                    output_segmentations_1,
                )
        return val_loss / (i + 1), loss.item()

    def get_loss_depth(self, epoch, output_depths, Y_depths, fx=0.1, epoch_threshold=10):
        if epoch < 1 and b_epoch_0_pure_segementation:
            return nn.MSELoss()(output_depths, Y_depths) * 0
        if epoch < epoch_threshold:
            return self.loss_depth(output_depths, Y_depths)
        else:
            return nn.MSELoss()(output_depths, Y_depths) * fx       # if fx is very small => loss_depth is always zero

    def j_loss(
        self,
        output_depths,
        Y_depths,
        output_segmentations,
        Y_segmentations,
        epoch=-1,
        ratio_threshold=0.4,

    ):
        # print("torch.max(output_depths)=", torch.max(output_depths))
        # print("torch.min(output_depths)=", torch.min(output_depths))
        ratio_out = get_ratio_out_raw(Y_segmentations, output_depths)
        loss_ratio_out = (1 - ratio_out)
        loss_seg = self.p.loss_segmentation_factor * self.loss_segmentation(
            output_segmentations, Y_segmentations
        )

        if self.p.loss_depth_in_factor == self.p.loss_depth_out_factor:
            loss_depth = 2.0 * self.p.loss_depth_in_factor * self.get_loss_depth(epoch, output_depths, Y_depths)
            # loss = loss_mse + loss_depth
            # return (
            #     loss
            #     if (loss < loss_depth * loss_mse_threshold_factor)
            #     else loss_depth * loss_mse_threshold_factor
            # )
        else:

            #
            idx = Y_depths > self.p.depth_datum
            idx_not = torch.logical_not(idx)
            loss_in = (
                self.p.loss_depth_in_factor
                * self.get_loss_depth(
                    epoch,
                    output_depths[idx_not].reshape(1, 1, -1),
                    Y_depths[idx_not].reshape(1, 1, -1),
                )
                if torch.count_nonzero(idx_not) > 0
                else 0
            )
            loss_out = (
                self.p.loss_depth_out_factor
                * self.get_loss_depth(
                    epoch,
                    output_depths[idx].reshape(1, 1, -1),
                    Y_depths[idx].reshape(1, 1, -1),
                )
                if torch.count_nonzero(idx) > 0
                else 0
            )
            loss_depth = loss_in + loss_out

        # print(f"loss_seg={loss_seg}")
        loss = loss_seg + loss_depth   # + loss_smoothness + loss_ssim
        fx = self.p.loss_coarse_threshold_factor / np.sqrt(max(epoch * self.p.loss_ratio_out_attenuation_factor, 1))
        fx_mininum = 0.5    # increase the value to reduce noise; decrease this value to get more depth details
        fx = max(fx, fx_mininum)
        print("....................fx=", fx)
        return (
            loss_seg
            * (1 + loss_depth / loss * fx),
            loss_seg, loss_ratio_out
        )

    def save_model(self):
        path_model = os.path.join(
            "models", self.p.model_name(), self.model.__class__.__name__
        )
        create_dir(path_model)
        model_fname = path_model + ".p"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_backbone_state_dict": self.optimizer_backbone.state_dict(),
                "optimizer_scratch_state_dict": self.optimizer_scratch.state_dict(),
            },
            model_fname,
        )
        print("Model saved at : {}".format(path_model))
        return model_fname

    def img_logger(
        self, X, Y_depths, Y_segmentations, output_depths, output_segmentations
    ):
        nb_to_show = (
            self.config["wandb"]["images_to_show"]
            if self.config["wandb"]["images_to_show"] <= len(X)
            else len(X)
        )
        tmp = X[:nb_to_show].detach().cpu().numpy()
        imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        if output_depths is not None:
            tmp = Y_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            depth_truths = np.repeat(tmp, 3, axis=1)
            tmp = output_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            # depth_preds = 1.0 - tmp
            depth_preds = tmp
        if output_segmentations is not None:
            tmp = Y_segmentations[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            segmentation_truths = np.repeat(tmp, 3, axis=1).astype("float32")
            tmp = torch.argmax(output_segmentations[:nb_to_show], dim=1)
            tmp = tmp.unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            segmentation_preds = tmp.astype("float32")
        # print("******************************************************")
        # print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
        # if output_depths != None:
        #     print(depth_truths.shape, depth_truths.mean().item(), depth_truths.max().item(), depth_truths.min().item())
        #     print(depth_preds.shape, depth_preds.mean().item(), depth_preds.max().item(), depth_preds.min().item())
        # if output_segmentations != None:
        #     print(segmentation_truths.shape, segmentation_truths.mean().item(), segmentation_truths.max().item(), segmentation_truths.min().item())
        #     print(segmentation_preds.shape, segmentation_preds.mean().item(), segmentation_preds.max().item(), segmentation_preds.min().item())
        # print("******************************************************")
        imgs = imgs.transpose(0, 2, 3, 1)
        if output_depths is not None:
            depth_truths = depth_truths.transpose(0, 2, 3, 1)
            depth_preds = depth_preds.transpose(0, 2, 3, 1)
        if output_segmentations is not None:
            segmentation_truths = segmentation_truths.transpose(0, 2, 3, 1)
            segmentation_preds = segmentation_preds.transpose(0, 2, 3, 1)
        output_dim = (
            int(self.config["wandb"]["im_w"]),
            int(self.config["wandb"]["im_h"]),
        )
#wandb.log()会将数据记录到当前的历史记录，换句话说就是每次运行到这里，系统就会将log内的参数值自动上传更新，一般数据会直接绘制成表格
# wandb.Image()用于图像的显示
        wandb.log(
            {
                "img": [
                    wandb.Image(
                        cv2.resize(im, output_dim), caption="img_{}".format(i + 1)
                    )
                    for i, im in enumerate(imgs)
                ]
            }
        )
        if output_depths is not None:
            wandb.log(
                {
                    "depth_truths": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="depth_truths_{}".format(i + 1),
                        )
                        for i, im in enumerate(depth_truths)
                    ],
                    "depth_preds": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="depth_preds_{}".format(i + 1),
                        )
                        for i, im in enumerate(depth_preds)
                    ],
                }
            )
        if output_segmentations is not None:
            wandb.log(
                {
                    "seg_truths": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="seg_truths_{}".format(i + 1),
                        )
                        for i, im in enumerate(segmentation_truths)
                    ],
                    "seg_preds": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="seg_preds_{}".format(i + 1),
                        )
                        for i, im in enumerate(segmentation_preds)
                    ],
                }
            )
