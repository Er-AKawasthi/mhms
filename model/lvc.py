import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False,
                 act_layer=nn.ReLU, groups=1, norm_layer=nn.BatchNorm2d,
                 drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

        c = out_channels

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if self.res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.act1(self.bn1(self.conv1(x)))
        if self.drop_block: x = self.drop_block(x)

        x2 = self.act2(self.bn2(self.conv2(x)))
        if self.drop_block: x2 = self.drop_block(x2)

        out = self.bn3(self.conv3(x2))
        if self.drop_block: out = self.drop_block(out)

        if self.drop_path:
            out = self.drop_path(out)

        if self.res_conv:
            residual = self.residual_bn(self.residual_conv(residual))

        out += residual
        out = self.act3(out)

        return (out, x2) if return_x_2 else out


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        self.in_channels = in_channels
        self.num_codes = num_codes

        std = 1. / ((num_codes * in_channels) ** 0.5)

        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels).uniform_(-std, std))
        self.scale = nn.Parameter(
            torch.empty(num_codes).uniform_(-1, 0))

    def scaled_l2(self, x, codewords, scale):
        B = x.size(0)
        N = x.size(1)
        C = self.in_channels
        K = self.num_codes

        expanded_x = x.unsqueeze(2).expand(B, N, K, C)
        reshaped_codewords = codewords.view(1, 1, K, C)
        reshaped_scale = scale.view(1, 1, K)

        scaled_l2 = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2

    def aggregate(self, assignment_weights, x, codewords):
        B, N, K = assignment_weights.size()
        C = codewords.size(1)

        expanded_x = x.unsqueeze(2).expand(B, N, K, C)
        reshaped_codewords = codewords.view(1, 1, K, C)
        assignment_weights = assignment_weights.unsqueeze(3)

        encoded = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2).contiguous()  # B x N x C
        weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        return self.aggregate(weights, x, self.codewords)  # B x K x C


class LVCBlock(nn.Module):
    """
    Lightweight Visual Context Block (LVC): context-aware attention
    """
    def __init__(self, in_channels, out_channels, num_codes):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes

        self.conv_1 = ConvBlock(in_channels, in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(in_channels, num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.LVC(x)  # B x C
        gamma = self.fc(enc)
        B, C, _, _ = x.size()
        gamma = gamma.view(B, C, 1, 1)
        return x * gamma
