
import torch

print(torch.cuda.is_available())  # cuda是否可用


print(torch.cuda.device_count())  # 返回GPU的数量


print(torch.cuda.get_device_name(0))  # 返回设备索引



print(torch.cuda.current_device())  # 返回当前设备索引

print(torch.rand(3, 3).cuda())