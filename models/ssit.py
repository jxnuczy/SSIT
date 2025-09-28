from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class SSFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.dlconv = nn.Conv2d(hidden_features,hidden_features, kernel_size=3, dilation=2, padding='same')

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class SCA(nn.Module):
    """global spectral attention (GSA)

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, dim, num_heads, bias):
        super(SCA, self).__init__()
        self.num_heads = num_heads
        self.kv = nn.Linear(dim, dim, bias)
        self.q = nn.Linear(dim, dim, bias)
        self.project_out = nn.Linear(dim, dim, bias)

    def forward(self, x):
        b, n, c = x.shape
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=2)
        q = self.q(x)

        q = q.reshape(-1, self.num_heads, q.shape[-1] // self.num_heads, n).permute(0, 1, 3, 2).contiguous()
        k = k.reshape(-1, self.num_heads, k.shape[-1] // self.num_heads, n).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(-1, self.num_heads, v.shape[-1] // self.num_heads, n).permute(0, 1, 3, 2).contiguous()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)



        # attn = (q @ k.transpose(-2, -1)) #* self.temperature
        attn = (k.transpose(-2, -1) @ q)

        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        # out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        out = out.reshape(b, n, c)
        out = self.project_out(out)
        return out

    def flops(self, patchresolution):
        flops = 0
        H, W, C = patchresolution
        flops += H * C * W * C
        flops += C * C * H * W
        return flops


class SPA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.permuted_window_size = (window_size[0] // 2, window_size[1] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.permuted_window_size[0])
        coords_w = torch.arange(self.permuted_window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size[1] - 1
        aligned_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        aligned_relative_position_index = aligned_relative_position_index.reshape \
            (self.permuted_window_size[0], self.permuted_window_size[1], 1, 1,
             self.permuted_window_size[0] * self.permuted_window_size[1]).repeat(1, 1, 2, 2, 1) \
            .permute(0, 2, 1, 3, 4).reshape(4 * self.permuted_window_size[0] * self.permuted_window_size[1],
                                            self.permuted_window_size[0] * self.permuted_window_size[1])  # FN*FN,WN*WN
        self.register_buffer('aligned_relative_position_index', aligned_relative_position_index)
        # compresses the channel dimension of KV
        self.kv = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # self.spectral_attn = GSA(dim,num_heads,bias=True)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        # compress the channel dimension of KV :(num_windows*b, num_heads, n//4, c//num_heads)
        kv = self.kv(x).reshape(b_, self.permuted_window_size[0], 2, self.permuted_window_size[1], 2, 2,
                                c // 4).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, n // 4, 2, self.num_heads,
                                                                             c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # keep the channel dimension of Q: (num_windows*b, num_heads, n, c//num_heads)
        q = self.q(x).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (num_windows*b, num_heads, n, n//4)

        relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1],
            -1)  # (n, n//4)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, n, n//4)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n // 4) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n // 4)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)  # 空间注意力

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.permuted_window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 1.5 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n / 4
        #  x = (attn @ v)
        flops += self.num_heads * n * n / 4 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops

class SSA(nn.Module):
    r"""  Transformer Block:Spatial-Spectral Multi-head self-Attention (SSMA)

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, drop_path=0.0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, bias=False):
        super(SSA, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = 0
        # if min(self.input_resolution) <= self.window_size:
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ssffn = SSFFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                          drop=drop)

        self.num_heads = num_heads
        self.spatial_attn = SPA(dim, [window_size, window_size], num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.spectral_attn = SCA(dim, num_heads, bias=True)


        self.spectral_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, dim, kernel_size=1)
        )

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # shifted_x = x.view(B, H, W, C)
        # c_x = x.view(B, C, H, W)

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        spatial_windows = self.spatial_attn(x_windows)
        spectral_windows = self.spectral_attn(x_windows)

        # merge windows
        spatial_windows = spatial_windows.view(-1, self.window_size, self.window_size, C)
        spectral_windows = spectral_windows.view(-1, self.window_size, self.window_size, C)

        # spatial_x ,spectral_x
        spectral_x = window_reverse(spectral_windows, self.window_size, H, W)  # (B,H,W,C)
        spatial_x = window_reverse(spatial_windows, self.window_size, H, W)
        spectral_x = spectral_x.view(B, C, H, W)
        spatial_x = spatial_x.view(B, C, H, W)

        spe_inter = self.spectral_interaction(spectral_x)  # 1,1,C
        spa_inter = self.spatial_interaction(spatial_x)  # H,W,1
        a1 = torch.sigmoid(spa_inter)
        a2 = torch.sigmoid(spe_inter)
        spa_x = spatial_x * a2
        spe_x = spectral_x * a1

        x = (spa_x + spe_x).view(B, H * W, C)



        # SSFFN
        x = shortcut + self.drop_path(x)
        x1 = self.norm2(x)
        x = x + self.drop_path(self.ssffn(x1, H, W))

        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class SSIT_Block(nn.Module):
    """
        residual spatial-spectral block (RSSB).
        Args:
            dim (int, optional): Embedding  dim of features. Defaults to 90.
            window_size (int, optional): window size of non-local spatial attention. Defaults to 8.
            depth (int, optional): numbers of Transformer block at this layer. Defaults to 6.
            num_head (int, optional):Number of attention heads. Defaults to 6.
            mlp_ratio (int, optional):  Ratio of mlp dim. Defaults to 2.
            qkv_bias (bool, optional): Learnable bias to query, key, value. Defaults to True.
            qk_scale (_type_, optional): The qk scale in non-local spatial attention. Defaults to None.
            drop_path (float, optional): drop_rate. Defaults to 0.0.
            bias (bool, optional): Defaults to False.
    """

    def __init__(self,
                 dim=90,
                 window_size=8,
                 depth=6,
                 num_head=6,
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 drop_path=0.1,
                 bias=False):
        super(SSIT_Block, self).__init__()

        self.smsblock = nn.Sequential(
            *[SSA(dim=dim, input_resolution=[64, 64], num_heads=num_head, window_size=window_size,
                  # shift_size=0 if (i % 2 == 0) else window_size // 2,
                  mlp_ratio=mlp_ratio,
                  drop_path=drop_path[i],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, bias=bias)
              for i in range(depth)])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        out = self.smsblock(x)

        out = self.conv(out) + x
        return out


class SSIT(nn.Module):
    """SST
     Spatial-Spectral Transformer for Hyperspectral Image Denoising

        Args:
            inp_channels (int, optional): Input channels of HSI. Defaults to 31.
            dim (int, optional): Embedding dimension. Defaults to 90.
            window_size (int, optional): Window size of non-local spatial attention. Defaults to 8.
            depths (list, optional): Number of Transformer block at different layers of network. Defaults to [ 6,6,6,6,6,6].
            num_heads (list, optional): Number of attention heads in different layers. Defaults to [ 6,6,6,6,6,6].
            mlp_ratio (int, optional): Ratio of mlp dim. Defaults to 2.
            qkv_bias (bool, optional): Learnable bias to query, key, value. Defaults to True.
            qk_scale (_type_, optional): The qk scale in non-local spatial attention. Defaults to None. If it is set to None, the embedding dimension is used to calculate the qk scale.
            bias (bool, optional):  Defaults to False.
            drop_path_rate (float, optional):  Stochastic depth rate of drop rate. Defaults to 0.1.
    """

    def __init__(self,
                 inp_channels=31,
                 dim=90,
                 window_size=[8, 8, 8, 8, 8, 8],
                 depths=[6, 6, 6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1
                 ):

        super(SSIT, self).__init__()

        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)  # shallow featrure extraction
        self.num_layers = depths
        self.layers = nn.ModuleList()
        # print(len(self.num_layers))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(self.num_layers)):
            layer = SSIT_Block(dim=dim,
                               window_size=window_size[i_layer],
                               depth=depths[i_layer],
                               num_head=num_heads[i_layer],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               # drop_path=0.1,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               bias=bias)
            self.layers.append(layer)

        self.output = nn.Conv2d(int(dim), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_delasta = nn.Conv2d(dim, inp_channels, 3, 1, 1)  # reconstruction from features

    def forward(self, inp_img):
        f1 = self.conv_first(inp_img)
        x = f1
        for layer in self.layers:
            x = layer(x)

        x = self.output(x + f1)
        x = self.conv_delasta(x) + inp_img
        return x


if __name__ == "__main__":
    x = torch.randn((2, 31, 128, 128))

    net = SSIT(inp_channels=31,
               dim=96,
               window_size=[32, ],
               depths=[6,  ],
               num_heads=[6, ],
               mlp_ratio=2,
               qkv_bias=True, qk_scale=None,
               bias=False,
               drop_path_rate=0.1)

    y = net(x)
    # 输出net的Parameter数量

    print(sum(p.numel() for p in net.parameters()))
    # 输出net的FLOPs数量
    print(y.shape)