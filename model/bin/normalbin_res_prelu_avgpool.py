import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary
import click
import os
sys.path.append(os.getcwd())
from model.bin.res_parts import ResBlock, Down, Up  # noqa:E402
from utils.torch_timer import TorchTimer  # noqa:E402


class DisparityRegression(nn.Module):
    def __init__(self, start, end, stride=1, dtype=torch.float32):
        super().__init__()

        start = start * stride
        end = end * stride
        self.disp = torch.arange(start, end, stride, out=torch.FloatTensor())
        self.disp = self.disp.view(1, -1, 1, 1).cuda()
        if dtype == torch.half:
            self.disp = self.disp.half()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out


class FeatureNetwork(nn.Module):
    def __init__(self, init_channels, down_levels, up_levels, block_size, binary):
        super(FeatureNetwork, self).__init__()

        activation = 'prelu'
        self.init_conv = nn.Sequential(nn.Conv2d(3, init_channels, 3, 1, 1),
                                       nn.BatchNorm2d(init_channels),
                                       nn.PReLU())
        self.top_conv = ResBlock(init_channels, init_channels, 3, stride=2, pad=1,
                                 binary=binary, bn=False, activation='')
        self.down_1 = nn.Sequential(Down(1, 2, n=1, binary=binary, bn=True, activation=activation,
                                         residual=True, pool='avg'),
                                    ResBlock(2, 2, 3, stride=1, pad=1,
                                             binary=binary, bn=False, activation=''))
        self.down_2 = nn.Sequential(Down(2, 4, n=1, binary=binary, bn=True, activation=activation,
                                         residual=True, pool='avg'),
                                    ResBlock(4, 4, 3, stride=1, pad=1,
                                             binary=binary, bn=False, activation=''))
        self.down_3 = nn.Sequential(Down(4, 8, n=1, binary=binary, bn=True, activation=activation,
                                         residual=True, pool='avg'),
                                    ResBlock(8, 8, 3, stride=1, pad=1,
                                             binary=binary, bn=False, activation=''))
        self.up_1 = Up(12, 4, n=1, binary=binary, bn=True, activation=activation,
                       residual=True)
        self.last_up = ResBlock(4, 4, 3, stride=1, pad=1,
                                binary=False, bn=False, activation='')

    def forward(self, x):
        out = self.init_conv(x)
        out = self.top_conv(out)
        out_1 = self.down_1(out)
        out_2 = self.down_2(out_1)
        out_3 = self.down_3(out_2)
        out_4 = self.up_1(out_3, out_2)
        return self.last_up(out_4)


class PostFeatureNetwork(nn.Module):
    def __init__(self, channels_3d, layers_3d, growth_rate, max_disp, binary):
        super(PostFeatureNetwork, self).__init__()
        self.channels_3d = channels_3d
        self.layers_3d = layers_3d
        self.growth_rate = growth_rate
        self.max_disp = max_disp

        activation = 'prelu'
        # Left processing
        self.down_1 = Down(self.max_disp, self.max_disp, n=1, binary=False, bn=True,
                           activation=activation, residual=True, pool='avg')
        self.down_conv_1 = ResBlock(self.max_disp, self.max_disp, ks=3, binary=binary,
                                    bn=True, activation=activation)
        self.down_conv_2 = ResBlock(self.max_disp, self.max_disp, ks=3, binary=binary, bn=False, activation='')

        self.up_1 = Up(self.max_disp * 2, self.max_disp, n=2, binary=binary, activation=activation,
                       bn=True, residual=True)
        self.last_conv = ResBlock(self.max_disp, self.max_disp, ks=3, binary=False, bn=False, activation='')

    def forward(self, x):
        out = self.down_1(x)
        out = self.down_conv_1(out)
        out = self.down_conv_2(out)
        out_up = self.up_1(out, x)
        return self.last_conv(out_up)


class NormalBinResPreluAvgpool(nn.Module):

    def __init__(self, max_disp=192, binary=False):
        super().__init__()
        self.down_levels = 3
        self.up_levels = 1
        self.init_channels = 1
        self.layers_3d = 4
        self.channels_3d = 4
        self.growth_rate = [4, 1, 1]
        self.block_size = 2
        factor = 2**(self.down_levels + 1) / 2**(self.up_levels)
        self.max_disp = int(max_disp // factor)

        self.feature_network = FeatureNetwork(init_channels=self.init_channels,
                                              down_levels=self.down_levels,
                                              up_levels=self.up_levels,
                                              block_size=self.block_size,
                                              binary=binary)
        self.cost_post = PostFeatureNetwork(self.channels_3d,
                                            self.layers_3d,
                                            self.growth_rate,
                                            self.max_disp,
                                            binary=binary)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def build_cost_volume(self, feat_l, feat_r, stride=1):
        if feat_l.dtype == torch.float32:
            cost = torch.zeros(feat_l.size()[0], self.max_disp // stride, feat_l.size()[2], feat_l.size()[3]).cuda()
        else:
            cost = torch.zeros(feat_l.size()[0], self.max_disp // stride, feat_l.size()[2], feat_l.size()[3]).cuda().half()

        for i in range(0, self.max_disp, stride):
            cost[:, i // stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i // stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i // stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

    def regression(self, cost, left):
        dtype = torch.float32
        if cost.dtype == torch.half:
            dtype = torch.half
        img_size = left.size()
        pred_low_res = DisparityRegression(0, self.max_disp, dtype=dtype)(F.softmax(-cost, dim=1))
        pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
        disp_up = F.interpolate(pred_low_res, (img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        return disp_up

    def forward(self, left, right):
        bs = left.size(0)
        left_and_right = torch.cat((left, right), 0)
        feats = self.feature_network.forward(left_and_right)
        l_feat = feats[0:bs, :, :, :]
        r_feat = feats[bs:bs * 2, :, :, :]

        # Cost volume pre
        cost = self.build_cost_volume(l_feat, r_feat)

        # Cost volume post processing
        cost_post = self.cost_post(cost)

        # Regression
        disp_up = self.regression(cost_post, left)

        if self.training:
            return disp_up, feats, cost, cost_post
        else:
            return disp_up


@click.command()
@click.option('--benchmark/--no-benchmark', default=False, help='Benchmark speed')
@click.option('--tensorrt/--no-tensorrt', default=False, help='Use tensorrt for benchmark')
def main(benchmark, tensorrt):
    # Print summary
    fsa = NormalBinResPreluAvgpool(max_disp=192).cuda()
    summary(fsa, [(3, 368, 1218), (3, 368, 1218)])
    if benchmark:
        print('Speed benchmark:')
        fsa.eval()
        tt = TorchTimer(times=200, warmup=10)
        torch.backends.cudnn.benchmark = True

        from torchvision import transforms
        # Data preparation
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

        from utils.dataloaders.kitti import KittiLoader
        kl = KittiLoader('kitti2012', '/opt/datasets/kitti2012/', training=False, validation=True, transform=transform)
        left, right, _ = kl[0]
        left = left.unsqueeze(0).cuda()
        right = right.unsqueeze(0).cuda()
        left_and_right = torch.cat((left, right), 0)

        if tensorrt:
            from torch2trt import torch2trt
            fsa.feature_network = torch2trt(fsa.feature_network, [left_and_right],
                                            fp16_mode=False, max_batch_size=2)
            feats = fsa.feature_network(left_and_right)
            l_feat = feats[0:1, :, :, :]
            r_feat = feats[1:2, :, :, :]
            cost = fsa.build_cost_volume(l_feat, r_feat)
            fsa.cost_post = torch2trt(fsa.cost_post, [cost],
                                      fp16_mode=False, max_batch_size=1)

        with torch.no_grad():

            # Full network
            full_mean, full_std, _ = tt.run(fsa, left, right)
            print(f'Full network elapsed mean time {full_mean:0.8f} s with std {full_std: 0.8f} s')
            print()

            # Convs
            conv_mean, conv_std, feats = tt.run(fsa.feature_network, left_and_right)
            print(f'Feature Conv elapsed mean time {conv_mean:0.8f} s with std {conv_std: 0.8f} s')

            # Cost volume
            l_feat = feats[0:1, :, :, :]
            r_feat = feats[1:2, :, :, :]
            cost_mean, cost_std, cost = tt.run(fsa.build_cost_volume, l_feat, r_feat)
            print(f'Cost elapsed mean time {cost_mean:0.8f} s with std {cost_std: 0.8f} s')

            # Post cost
            post_cost_mean, post_cost_std, proccesed_cost = tt.run(fsa.cost_post, cost)
            print(f'Post Cost elapsed mean time {post_cost_mean:0.8f} s with std {post_cost_std: 0.8f} s')

            # Regression
            r_mean, r_std, out = tt.run(fsa.regression, cost, left)
            print(f'Regression elapsed mean time {r_mean:0.8f} s with std {r_std: 0.8f} s')

            # Total time by parts
            total = conv_mean + cost_mean + post_cost_mean + r_mean
            print(f'Total summing means {total}')


if __name__ == "__main__":
    main()
