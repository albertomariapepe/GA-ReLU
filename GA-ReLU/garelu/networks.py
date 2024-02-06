import torch.nn.functional as F
import torch
from torchsummary import summary
from acts import GArelu
from cliffordlayers.models.basic.twod import (
    CliffordFluidNet2d,
    CliffordBasicBlock2d,
    CliffordFourierBasicBlock2d,
    BasicBlock,
    ResNet as RN
)
from cliffordlayers.models.utils import partialclass
import neuralop
from neuralop.models import FNO


def FourierNet(in_channels, out_channels):
    return FNO(n_modes=(16, 16), hidden_channels=128, in_channels=in_channels*3, out_channels= out_channels*3, activation = F.relu)

in_channels = 2
out_channels = 1


#note that the shape of the input data is slightly different than for Clifford Networks, i.e.
#(B, in*3, x_size, y_size) 
#this is because Conv2d accepts only 4D tensors
#Clifford Networks need data with shape (B, in, x_size, y_size, 3), i.e. they must be 5D tensors
def ResNet(in_channels, out_channels):
    return RN(img_channels=in_channels*3, out_channels=out_channels*3, num_layers = 18, block=BasicBlock)


'''
model = FourierNet(in_channels*3, out_channels*3)
print(model)
x = torch.randn(8, 3*in_channels, 128, 128)
y = model(x)
print(y.shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
#checking that the number of parameters of the adapted ResNet18 matches those of Brandstetter et al.
#about 2.4M, which is consistent
'''
'''
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
'''

def CliffordResNet(in_channels, out_channels, activation):
    if activation == 3:
        act = GArelu
    else:
        act = F.relu
    
    model = CliffordFluidNet2d(
        g=[1, 1],
        block=CliffordBasicBlock2d,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        activation=act,
        norm=False,
        rotation=False,
    )

    return model



def CliffordResNetNorm(in_channels, out_channels, activation):
    if activation == 3:
        act = GArelu
    else:
        act = F.relu
    
    model = CliffordFluidNet2d(
        g=[1, 1],
        block=CliffordBasicBlock2d,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        activation=act,
        norm=True,
        rotation=False,
    )
    
    return model

def CliffordResNetRot(in_channels, out_channels):
    model = CliffordFluidNet2d(
        g=[-1, -1],
        block=CliffordBasicBlock2d,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        activation=F.gelu,
        norm=True,
        rotation=True,
    )
    
    return model

def CliffordFourierNet(in_channels, out_channels, activation):
    if activation == 3:
        act = GArelu
    else:
        act = F.relu

    model = CliffordFluidNet2d(
        g=[1, 1],
        block=partialclass("CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=16, modes2=16),
        num_blocks=[1, 1, 1, 1],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=48,
        activation=act,
        norm=False,
        rotation=False,
    )
    return model


def CliffordFourierNetNorm(in_channels, out_channels):
    model = CliffordFluidNet2d(
        g=[1, 1],
        block=partialclass("CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=16, modes2=16),
        num_blocks=[1, 1, 1, 1],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        activation=F.gelu,
        norm=True,
        rotation=False,
    )

    return model


def CliffordFourierNetRot(in_channels, out_channels):
    model = CliffordFluidNet2d(
        g=[-1, -1],
        block=partialclass("CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=32, modes2=32),
        num_blocks=[1, 1, 1, 1],
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        activation=F.gelu,
        norm=True,
        rotation=True,
    )
    return model


