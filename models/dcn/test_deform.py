from deform_conv import DeformConv
import torch
conv = DeformConv(10,20,3,padding=1)
conv.cuda()
x = torch.rand([1,10,416,416]).cuda()
offset = torch.rand([1,2*9,416,416]).cuda()
y = conv(x,offset)
print(y.shape)
