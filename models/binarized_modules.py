import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function


class SignSTE(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input.ge(-1) & input.le(1)
        grad_input = torch.where(mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        inputs = SignSTE.apply(inputs)
        self.weight_bin_tensor = SignSTE.apply(self.weight)
        self.weight_bin_tensor.requires_grad_()

        out = nn.functional.linear(inputs, self.weight_bin_tensor)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.xx = []
        self.yy = []
        self.influence_state = torch.from_numpy(np.zeros((kargs[1], kargs[1])))
        self.feature = None

    def forward(self, inputs):
        inputs = SignSTE.apply(inputs)
        self.weight_bin_tensor = SignSTE.apply(self.weight)
        self.weight_bin_tensor.requires_grad_()

        out = nn.functional.conv2d(inputs, self.weight_bin_tensor, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        self.feature = out
        # rl
        length = len(self.xx)
        for i in range(length):
            out[:, self.yy[i], :, :] = fp_add()(out[:, self.yy[i], :, :], out[:, self.xx[i], :, :],
                                                self.influence_state[self.xx[i]][self.yy[i]])

        return out

    def set(self, xx, yy, inf_state):
        self.xx = xx
        self.yy = yy
        self.influence_state = torch.from_numpy(inf_state).float()


class fp_add(torch.autograd.Function):
    def forward(ctx, x_original, dep_feature, alpha):
        ctx.save_for_backward(x_original)
        size = x_original.size()
        m_1 = torch.FloatTensor([200]).expand(size).cuda()
        m_2 = torch.FloatTensor([300]).expand(size).cuda()
        m_3 = torch.FloatTensor([250]).expand(size).cuda()
        m_4 = torch.FloatTensor([200]).expand(size).cuda()

        n_1 = torch.FloatTensor([-200]).expand(size).cuda()
        n_2 = torch.FloatTensor([-250]).expand(size).cuda()
        n_3 = torch.FloatTensor([-300]).expand(size).cuda()
        n_4 = torch.FloatTensor([-300]).expand(size).cuda()
        # print(np.max(dep_feature.data.cpu().numpy()))
        # alpha *= 5
        # print(alpha)
        # if alpha == 0.0007:
        #     x_original_ = x_original + ((torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_2).float().cuda()\
        #                 + torch.ge(dep_feature, m_3).float().cuda() + torch.ge(dep_feature, m_4).float().cuda()) - \
        #                 (torch.le(dep_feature, n_1).float().cuda() + torch.le(dep_feature, n_2).float().cuda()\
        #                 + torch.le(dep_feature, n_3).float().cuda() + torch.le(dep_feature, n_4).float().cuda()))*2
        # elif alpha == 0.0005:
        #     x_original_ = x_original + ((torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_2).float().cuda() + torch.ge(dep_feature, m_3).float().cuda())\
        #                     - (torch.le(dep_feature, n_1).float().cuda() + torch.le(dep_feature, n_2).float().cuda() + torch.le(dep_feature, n_3).float().cuda()))*2
        # elif alpha == 0.0003:
        #     x_original_ = x_original + ((torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_2).float().cuda())\
        #                     - (torch.le(dep_feature, n_1).float().cuda() + torch.le(dep_feature, n_2).float().cuda()))*2
        if alpha == 0.04:
            x_original_ = x_original + (
                        torch.ge(dep_feature, m_1).float().cuda() - torch.le(dep_feature, n_1).float().cuda()) * 100
        elif alpha == 0:
            x_original_ = x_original
        elif alpha == -0.04:
            x_original_ = x_original - (
                        torch.ge(dep_feature, m_1).float().cuda() + torch.le(dep_feature, n_1).float().cuda()) * 100
        # elif alpha == -0.0003:
        #     x_original_ = x_original - ((torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_2).float().cuda())\
        #                     + (torch.le(dep_feature, n_1).float().cuda() + torch.le(dep_feature, n_2).float().cuda()))*2
        # elif alpha == -0.0005:
        #     x_original_ = x_original - ((torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_2).float().cuda() + torch.ge(dep_feature, m_3).float().cuda())\
        #                     + (torch.le(dep_feature, n_1).float().cuda() + torch.le(dep_feature, n_2).float().cuda() + torch.le(dep_feature, n_3).float().cuda()))*2
        # elif alpha == -0.0007:
        #     x_original_ = x_original - ((torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_2).float().cuda() \
        #                   + torch.ge(dep_feature, m_1).float().cuda() + torch.ge(dep_feature, m_4).float().cuda())\
        #                     + (torch.le(dep_feature, n_1).float().cuda() + torch.le(dep_feature, n_2).float().cuda()\
        #                 + torch.le(dep_feature, n_3).float().cuda() + torch.le(dep_feature, n_4).float().cuda()))*2
        return x_original_

    def backward(ctx, grad_output):
        return grad_output, None, None
