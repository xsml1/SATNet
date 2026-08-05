# -----------------------------------------------------------------------------
# SATNet: Structure-Aware Transformer Network for Underwater Image Enhancement
# -----------------------------------------------------------------------------
# This code is built upon the U-shaped Transformer architecture of Restormer:
#   S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang,
#   "Restormer: Efficient Transformer for High-Resolution Image Restoration,"
#   CVPR 2022.
# Restormer is released under the MIT License:
#   https://github.com/swz30/Restormer
# -----------------------------------------------------------------------------
"""Structure-Aware Transformer Network (SATNet) for underwater image enhancement.

The architecture is a U-shaped encoder-decoder built upon the Restormer
transformer backbone. It introduces a learnable Sobel convolution (LSConv) for
structure extraction, a Structure-guided Cascaded Transformer Block (SCTB)
with cascaded attention and adaptive layer normalization, and a
Structure-guided Dynamic Upsampling (SDU) module in the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias   = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: [B,C,H,W] -> [B,H,W,C]
        x_ = x.permute(0, 2, 3, 1)
        x_ = F.layer_norm(x_, (x_.shape[-1],), self.weight, self.bias, self.eps)
        return x_.permute(0, 3, 1, 2)

class AdaLayerNorm2d(nn.Module):
    """
    条件：全局 mean/std
    输出：LN(x) * (1 + gamma) + beta
    """
    def __init__(self, dim, r=4, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.norm = LayerNorm2d(dim, eps=eps, affine=False)

        hidden = max(dim // r, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim * 2, 1),
        )
        # identity init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        var = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=(2, 3), keepdim=True)

        # 2. 更加安全的开根号
        std = torch.sqrt(var + self.eps)
        stat = torch.cat([mean, std], dim=1)        # [B,2C,1,1]
        gamma_beta = self.mlp(stat)                 # [B,2C,1,1]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = torch.tanh(gamma)
        x_n = self.norm(x)
        return x_n * (1.0 + gamma) + beta


class LearnableSobelEdge(nn.Module):
    def __init__(self):
        super().__init__()

        # self.sobel = nn.Conv2d(3,2,3,padding=1, padding_mode='reflect',bias=False)
        self.sobel = nn.Conv2d(3, 2, 3, padding=1, padding_mode='replicate', bias=False)
        # self.sobel = nn.Conv2d(3, 2, 3, padding=1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

        self._init_sobel()

    def _init_sobel(self):

        sobel_x = torch.tensor(
            [[-1,0,1],[-2,0,2],[-1,0,1]],
            dtype=torch.float32)

        sobel_y = torch.tensor(
            [[-1,-2,-1],[0,0,0],[1,2,1]],
            dtype=torch.float32)

        weight = torch.zeros(2,3,3,3)

        for c in range(3):
            weight[0,c] = sobel_x
            weight[1,c] = sobel_y

        self.sobel.weight.data = weight

    def forward(self,x):

        grad = self.sobel(x)

        gx = torch.abs(grad[:,0:1])
        gy = torch.abs(grad[:,1:2])

        edge = self.alpha*gx + self.beta*gy

        return edge
##########################################################################
class EdgeGuidedDynamicConv(nn.Module):

    def __init__(self, dim, ksize=3):
        super().__init__()

        self.ksize = ksize
        self.channels = dim
        self.emb = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1)
        # edge -> dynamic kernel
        self.kernel_gen = nn.Conv2d(
            dim,
            dim * ksize * ksize,
            kernel_size=1
        )

    def forward(self, feat, edge):

        B, C, H, W = feat.shape
        k = self.ksize
        pad = (k - 1) // 2

        # generate kernel
        kernel = self.kernel_gen(self.emb(edge))

        # reshape kernel
        kernel = kernel.view(B, C, k*k, H, W)
        kernel = F.softmax(kernel, dim=2)

        # padding
        feat_pad = F.pad(feat, (pad, pad, pad, pad), mode='replicate')

        # unfold
        feat_patch = F.unfold(feat_pad, kernel_size=k)

        feat_patch = feat_patch.view(B, C, k*k, H, W)

        # dynamic filtering
        out = torch.sum(feat_patch * kernel, dim=2)

        return out

class Cascaded_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cascaded_Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.temperature_2 = nn.Parameter(torch.ones(self.num_heads, 1, 1))


        self.project_in = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Softmax(dim=-1)

        self.edge_inj = EdgeGuidedDynamicConv(dim=dim)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False)

        self.q2_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q2_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)


    def forward(self, x, edge):
        # edge = torch.randn((1, 1, 256, 256)).cuda() #测试参数量时候用
        b, c, h, w = x.shape

        x1, x2 = self.project_in(x).chunk(2, dim=1)

        # channel-wise attention
        qv = self.dwconv(x1) * x2
        q, v = qv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out1 = (attn @ v)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) + x
        # edge injection
        q_2 = self.q2_2(self.q2_1(out1))

        kv_edge = self.edge_inj(out1, edge)
        kv_edge = self.kv_conv(self.kv(kv_edge))
        k_edge, v_edge = kv_edge.chunk(2, dim=1)
        q_2 = rearrange(q_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_edge = rearrange(k_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_edge = rearrange(v_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_2 = torch.nn.functional.normalize(q_2, dim=-1)
        k_edge = torch.nn.functional.normalize(k_edge, dim=-1)

        attn = (q_2 @ k_edge.transpose(-2, -1)) * self.temperature_2
        attn = attn.softmax(dim=-1)

        out = (attn @ v_edge)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x




class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = AdaLayerNorm2d(channels)
        self.attn = Cascaded_Attention(channels, num_heads, bias=False)
        self.norm2 = AdaLayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, inp):
        x, edge = inp[0], inp[1]

        if edge.shape[2] != x.shape[2]:
            edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        x = x + self.attn(self.norm1(x), edge)
        x = x + self.ffn(self.norm2(x))

        return [x, edge]


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)



class EdgeGuidedUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(EdgeGuidedUpsample, self).__init__()
        self.scale_factor = scale_factor

        # 偏移量预测分支：基于特征图预测采样点的微调方向
        # 输出通道为 2，分别代表 (delta_x, delta_y)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )

        # 初始化为0，保证初期模型表现接近标准双线性插值
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

        self.reduce = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0)

    def forward(self, x, edge_map):
        """
        x: 低分辨率特征图 [B, C, H, W]
        edge_map: 高分辨率边缘图 [B, 1, H*scale, W*scale]
        """
        B, C, H, W = x.shape
        target_h, target_w = H * self.scale_factor, W * self.scale_factor

        # 1. 将低分辨率特征图初步放大（作为基础背景），并将edge map采样至与x_upsampled相同分辨率
        x_upsampled = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        edge_map = F.interpolate(edge_map, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # 2. 预测采样偏移量
        # 我们将初步放大的特征和边缘图拼接，让模型感知 哪里需要修正
        feat_with_edge = torch.cat([x_upsampled, edge_map], dim=1)
        offset = self.offset_conv(feat_with_edge)  # [B, 2, target_h, target_w]

        # 3. 利用边缘图对偏移量进行调制
        # 只有在 edge_map 强度高的地方，offset 才会起明显作用
        # 这可以保证平滑区域不被打乱，
        controlled_offset = offset * edge_map

        # 4. 生成采样网格 (Grid)
        # 生成标准归一化坐标网格 [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, target_h),
            torch.linspace(-1, 1, target_w),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).to(x.device)  # [target_h, target_w, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, target_h, target_w, 2]

        # 将预测的 offset 叠加到标准网格上
        # 注意：offset 需要根据尺度进行缩放归一化，这里简写为直接相加
        # 实际上 offset 的量级应该与像素间距对应
        multiplier = 2.0 / max(target_h, target_w)
        final_grid = grid + controlled_offset.permute(0, 2, 3, 1) * multiplier

        # 5. 执行重采样
        # 从原始低分辨率特征图中，根据引导后的坐标取值
        out = F.grid_sample(x, final_grid, mode='bilinear', padding_mode='border', align_corners=False)

        return self.reduce(out)
class UpSample1(nn.Module):
    def __init__(self, channels):
        super(UpSample1, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)  





class UpS(nn.Module):
    def __init__(self, channels):
        super(UpS, self).__init__()
        self.Eups = EdgeGuidedUpsample(channels)
        self.Sups = UpSample1(channels)
        self.reduce = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)

    def forward(self, x, edge):
        out = torch.cat([self.Eups(x, edge), self.Sups(x)], dim=1)
        # print(out.shape)
        return self.reduce(out)



class SATNet(nn.Module):
    """Structure-Aware Transformer Network.

    FLOPs: ~24.48 G, Params: ~3.60 M (at 256 x 256 input).
    """

    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[64, 32, 16, 64]):
        super(SATNet, self).__init__()

        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.learned_edge = LearnableSobelEdge()
        self.edge_emb = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                                      nn.GELU(),
                                      nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1))
        self.encoders = nn.ModuleList(
            [nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for
             num_tb, num_ah, num_ch in
             zip(num_blocks, num_heads, channels)])

        self.down1 = DownSample(channels[0])
        self.down2 = DownSample(channels[1])
        self.down3 = DownSample(channels[2])
        self.ups_1 = UpS(128)
        self.ups_2 = UpS(64)
        self.ups_3 = UpS(32)

        self.reduces2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.reduces1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders.append(nn.Sequential(
            *[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)
        self.outputl = nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)


    def forward(self, RGB_input, is_train = False):
        pred_edge = self.learned_edge(RGB_input)
        edge_emb = self.edge_emb(pred_edge)
        ###-------encoder for RGB-------####
        fo_rgb = self.embed_conv_rgb(RGB_input)
        out_enc_rgb1 = self.encoders[0]([fo_rgb, edge_emb])[0]
        out_enc_rgb2 = self.encoders[1]([self.down1(out_enc_rgb1), edge_emb])[0]
        # print(out_enc_rgb2.shape)

        out_enc_rgb3 = self.encoders[2]([self.down2(out_enc_rgb2), edge_emb])[0]
        # print(out_enc_rgb3.shape)
        out_enc_rgb4 = self.encoders[3]([self.down3(out_enc_rgb3), edge_emb])[0]
        # print(out_enc_rgb4.shape)

        ###-------Dencoder------###
        out_dec3 = self.decoders[0]([self.reduces1(torch.cat([(self.ups_1(out_enc_rgb4, edge_emb)), out_enc_rgb3], dim=1)), edge_emb])[0]
        # print(out_dec3.shape)
        out_dec2 = self.decoders[1]([self.reduces2(torch.cat([self.ups_2(out_dec3, edge_emb), out_enc_rgb2], dim=1)), edge_emb])[
            0]
        # print(out_dec2.shape)
        fd = self.decoders[2]([torch.cat([self.ups_3(out_dec2, edge_emb), out_enc_rgb1], dim=1), edge_emb])[0]

        fr = self.refinement([fd, edge_emb])[0]
        if is_train:
            # 训练mode， 输出预测的边缘，与gt的边缘做损失
            return self.output(self.outputl(fr)), pred_edge
        else:
            return self.output(self.outputl(fr))
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SATNet().to(device)
    img = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(img)
        print('Output shape:', out.shape)
    try:
        from ptflops import get_model_complexity_info

        flops, params = get_model_complexity_info(
            model, (3, 256, 256), as_strings=False, print_per_layer_stat=False
        )
        print(f'FLOPs: {flops / 1e9:.4f} G, Params: {params / 1e6:.4f} M')
    except ImportError:
        print('ptflops not installed; skip FLOPs estimation.')
