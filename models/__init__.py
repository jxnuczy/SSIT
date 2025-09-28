from .competing_methods import *
from .sert import SERT
# from .hsy_ssit import SSIT
# from models.my_model_test.our_model import SSTR
# from models.my_model_test.new_model_1221 import SSI
# from .mamba import Sr_Spe
# from .Umamba import Restormer
from .my_model import *
from .TDSAT import TDSAT

from .onlyspa import SSIT

def spasca():
    net = SSIT(inp_channels=31,
                   dim=96,
                   window_size=[32, 32, 32, 32],
                   depths=[6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6],
                   mlp_ratio=2,
                   qkv_bias=True, qk_scale=None,
                   bias=False,
                   drop_path_rate=0.1)

    net.use_2dconv = True
    net.bandwise = False
    return net

def spasca_real():
    net = SSIT(inp_channels=34,
                   dim=96,
                   window_size=[32, 32, 32, 32],
                   depths=[6, 6, 6, 6],
                   num_heads=[6, 6, 6, 6],
                   mlp_ratio=2,
                   qkv_bias=True, qk_scale=None,
                   bias=False,
                   drop_path_rate=0.1)

    net.use_2dconv = True
    net.bandwise = False
    return net

def tdsat():
    net = TDSAT(1, 16, 5, [1, 3])

    net.use_2dconv = False
    net.bandwise = False
    return net
def sca():
    net = Sr_Spe(inp_channels=31,
                 dim=96,
                 window_size=[32, 32, 32, 32, ],
                 depths=[6, 6, 6, 6, ],
                 num_heads=[6, 6, 6, 6, ],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1)
    net.use_2dconv = True
    net.bandwise = False
    return net

def spa():
    net = Sr_Spe(inp_channels=31,
                 dim=96,
                 window_size=[32, 32, 32, 32],
                 depths=[6, 6, 6, 6, ],
                 num_heads=[6, 6, 6, 6, ],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1)
    net.use_2dconv = True
    net.bandwise = False
    return net

# def ssit():
#     net = SSIT(inp_channels=31,
#          dim=96,
#          window_size=[32, 32,32,32 ],
#          depths=[6, 6,6,6  ],
#          num_heads=[6, 6,6,6  ],
#          mlp_ratio=2,
#          qkv_bias=True, qk_scale=None,
#          bias=False,
#          drop_path_rate=0.1)
#
#     net.use_2dconv = True
#     net.bandwise = False
#     return net


# def mamba():
#     net = Sr_Spe(inp_channels=31,
#                  dim=24,
#                  window_size=[8, 16, 32,],
#                  depths=[4, 4, 4,],
#                  mlp_ratio=2,
#                  bias=False,
#                  drop_path_rate=0.1)
#
#     net.use_2dconv = True
#     net.bandwise = False
#
#     return net

# def gca():
#     net = Sr_Spe(inp_channels=31,
#                  dim =96,
#                  window_size=[32, 32,],
#                  depths=[6, 6,],
#                  num_heads=[6,6,],
#                  mlp_ratio=2,
#                  qkv_bias=True, qk_scale=None,
#                  bias=False,
#                  drop_path_rate=0.1)
#
#     net.use_2dconv = True
#     net.bandwise = False
#
#     return net

# def slstm2():
#     net = VisionLSTM2(dim=96,
#                       inp_channels=31,
#                       input_shape=(31, 128, 128),
#                       patch_size=16,
#                       depth=6,
#                       pooling="bilateral_flatten",
#                       drop_path_rate=0.0,
#                       drop_path_decay=False,
#                       stride=None,
#                       legacy_norm=False,
#                       conv_kind="2d",
#                       conv_kernel_size=3,
#                       proj_bias=True,
#                       norm_bias=True)
#     net.use_2dconv = True
#     net.bandwise = False
#     return net
#
# def ukan():
#     net = UKAN(num_classes=31,
#                  input_channels=31,
#                  deep_supervision=False,
#                  img_size=128,
#                  patch_size=16,
#                  in_chans=31,
#                  embed_dims=[256, 96, 96],
#                  no_kan=False,
#                  drop_rate=0.,
#                  drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm,
#                  depths=[1, 1, 1])
#     net.use_2dconv = True
#     net.bandwise = False
#     return net
#
# def slstm():
#     net = VisionLSTM(dim=96,
#                     input_shape=(31, 128, 128),
#                     inp_channels=31,
#                     patch_size=8,
#                     depth=6,
#                     pooling=None,
#                     drop_path_rate=0.0,
#                     stride=None,
#                     alternation="bidirectional",
#                     drop_path_decay=False,
#                     legacy_norm=False)
#     net.use_2dconv = True
#     net.bandwise = False
#     return net
#
#
# def sslstm():
#     net = SSLSTM(inp_channels=31,
#            dim=96,
#            window_size=[32, ],
#            depths=[6, ],
#            mlp_ratio=2,
#            qkv_bias=True, qk_scale=None,
#            drop_path_rate=0.1)
#     net.use_2dconv = True
#     net.bandwise = False
#     return net






def ssit_urban():
    net = SSIT(inp_channels=210,
                 dim =96,
                 window_size=[16, 16,16],
                 depths=[6, 6,6],
                 num_heads=[6, 6,6 ],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1)

    net.use_2dconv = True
    net.bandwise = False

    return net


def srspe():
    net = Sr_Spe(inp_channels=31,
                 dim =96,
                 window_size=[32, 32,32,32],
                 depths=[6, 6,6,6],
                 num_heads=[6, 6,6,6 ],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1)

    net.use_2dconv = True
    net.bandwise = False

    return net

def srspe_hou():
    net = Sr_Spe(inp_channels=46,
                 dim =96,
                 window_size=[32, 32, 32, 32],
                 depths=[6, 6, 6, 6,],
                 num_heads=[6, 6, 6, 6],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1)

    net.use_2dconv = True
    net.bandwise = False

    return net

def sert_base():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32

    net.use_2dconv = True
    net.bandwise = False
    return net

def sert_tiny():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32] ,        depths=[ 4,4],         num_heads=[ 6,6],split_sizes=[2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
    net.use_2dconv = True
    net.bandwise = False
    return net

def sert_small():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 4,4,4],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
    net.use_2dconv = True
    net.bandwise = False
    return net

def sert_urban():
    net = SERT(inp_channels=210,dim = 96*2,         window_sizes=[8,16,16] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[2,4,4],mlp_ratio=2,down_rank=8,memory_blocks=128)
    net.use_2dconv = True
    net.bandwise = False
    return net


def sert_real():
    net = SERT(inp_channels=34,dim = 96,         window_sizes=[16,32,32] ,        depths=[6,6,6],down_rank=8,         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,memory_blocks=64)

    net.use_2dconv = True
    net.bandwise = False
    return net

def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net

def grn_net():
    net = U_Net_GR(in_ch=31,out_ch=31)
    net.use_2dconv = True
    net.bandwise = False
    return net


def grn_net_real():
    net = U_Net_GR(in_ch=34,out_ch=34)
    net.use_2dconv = True
    net.bandwise = False
    return net

def grn_net_urban():
    net = U_Net_GR(in_ch=210,out_ch=210)
    net.use_2dconv = True
    net.bandwise = False
    return net


def t3sc():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_real():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_real.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_urban():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_urban.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def macnet():
    net = MACNet(in_channels=1,channels=16,num_half_layer=5)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst():
    net = SST(inp_channels=31,dim = 90,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_real():
    net = SST(inp_channels=34,depths=[6,6,6])
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_urban():
    net = SST(inp_channels=210,dim = 210,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net