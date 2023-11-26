import torch
import torch.nn as nn
from torchmetrics.functional import image_gradients


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    # print("prediction.shape=", prediction.shape, prediction.ndim)
    # print("target.shape=", target.shape, target.ndim)
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


# def get_smooth_loss(disp, img):
def smoothness_loss(prediction, target, reduction=reduction_batch_based):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    img = target
    disp = prediction
    grad_disp_x = torch.abs(disp[:, :, :-1] - disp[:, :, 1:])
    grad_disp_y = torch.abs(disp[:, :-1, :] - disp[:, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :-1, :] - img[:, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


# def smoothness_loss(prediction, target, reduction=reduction_batch_based):
#     mask = target > 0
#     M = torch.sum(mask, (1, 2))
#     target = target[None, :]
#     prediction = prediction[None, :]
#     dy_true, dx_true = image_gradients(target)
#     dy_pred, dx_pred = image_gradients(prediction)
#     weights_x = torch.exp(-torch.mean(torch.abs(dx_true)))
#     weights_y = torch.exp(-torch.mean(torch.abs(dy_true)))
#     # Depth smoothness
#     smoothness_x = dx_pred * weights_x * mask
#     smoothness_y = dy_pred * weights_y * mask
#     depth_smoothness_loss = torch.abs(smoothness_x) + torch.abs(smoothness_y)
#     return reduction(depth_smoothness_loss, M)


class MSELoss(nn.Module):  # used by ScaleAndShiftInvariantLoss
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class SmoothnessLoss(MSELoss):
    def __init__(self, reduction="batch-based"):
        super().__init__()
        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target):
        return smoothness_loss(prediction, target, reduction=self.__reduction)


class GradientLoss(nn.Module):  # used by ScaleAndShiftInvariantLoss
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


unknown_factor = 1e-3


class CeAndMse(nn.Module):
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    def __init__(self, wt_ce=0.5):
        super().__init__()
        self.wt_ce = wt_ce

    def forward(self, *args, **kwargs):
        y0 = self.ce.forward(*args, **kwargs)
        y1 = self.mse.forward(*args, **kwargs)
        return self.wt_ce * y0 * unknown_factor + (1.0 - self.wt_ce) * y1

#规格和平移不变量损失



class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target):
        # preprocessing
        mask = target > 0

        # calculate
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        # print(f"scale={scale.item()}, shift={shift.item()}.....")

        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target, mask
            )

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
