<<<<<<< Updated upstream
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation,
            normalization):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        if normalization is not None:
            self.normalization = normalization(out_channels)
        else:
            self.normalization = nn.Identity()
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        return self.activation(x)


class ResBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size):
        super(ResBlock, self).__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.PReLU,
            normalization=nn.BatchNorm2d)

        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.Identity,
            normalization=nn.BatchNorm2d)

    def _residual(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return x + self._residual(x)


class SISR_Resblocks(nn.Module):
    def __init__(self, num_blocks):
        super(SISR_Resblocks, self).__init__()

        self.resblocks = []
        for i in range(num_blocks):
            self.resblocks.append(
                ResBlock(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3))
        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x):
        return self.resblocks(x)


class Generator(nn.Module):

    def __init__(self, resblocks):
        super(Generator, self).__init__()

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.PReLU,
            normalization=None)

        self.conv2 = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.Identity,
            normalization=nn.BatchNorm2d)

        self.resblocks = resblocks

        self.conv3 = Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.Identity,
            normalization=None)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.prelu = nn.PReLU()

        self.conv4 = Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=9,
            stride=1,
            padding=9//2,
            activation=nn.Tanh,
            normalization=None)

    def forward(self, x):
        skip = self.conv1(x)
        x = self.resblocks(skip)
        x = self.conv2(x) + skip
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.conv4(x)
        return x
=======
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

def init_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

class Conv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            normalization,
            activation=None):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        # initialize our weights to be very low since this is a very deep network
        self.conv.apply(init_weights)

        if normalization is not None:
            self.normalization = normalization(out_channels)
        else:
            self.normalization = nn.Identity()
        
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size):
        super(ResBlock, self).__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.PReLU,
            normalization=nn.BatchNorm2d)

        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.Identity,
            normalization=nn.BatchNorm2d)

    def _residual(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return x + self._residual(x)

# as defined by the ESRGAN paper
# TODO: make the lnumber of residuable blocks configurable
class ResBlockDense(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            growth_channels,
            kernel_size):
        super(ResBlockDense, self).__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            #out_channels=out_channels,
            out_channels=growth_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        self.conv2 = Conv2d(
            #in_channels=in_channels * 2,
            in_channels=in_channels + (growth_channels),
            #out_channels=out_channels,
            out_channels=growth_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        self.conv3 = Conv2d(
            #in_channels=in_channels * 3,
            in_channels=in_channels + (growth_channels*2),
            #out_channels=out_channels,
            out_channels=growth_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        self.conv4 = Conv2d(
            #in_channels=in_channels * 4,
            in_channels=in_channels + (growth_channels*3),
            #out_channels=out_channels,
            out_channels=growth_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            activation=nn.LeakyReLU,
            normalization=None)
        # The last concat convolution before we leave the resblock
        self.conv5 = Conv2d(
            #in_channels=in_channels * 5,
            in_channels=in_channels + (growth_channels*4),
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            #activation=nn.Identity,
            normalization=None)

    #def _residual(self, x):
    #    skip = x
    #    c1 = self.conv1(x)
    #    c2 = self.conv2(torch.cat([skip, c1], 1))
    #    c3 = self.conv3(torch.cat([skip, c1, c2], 1))
    #    c4 = self.conv4(torch.cat([skip, c1, c2, c3], 1))
    #    c5 = self.conv5(torch.cat([skip, c1, c2, c3, c4], 1))
    #
    #    return c5

    def forward(self, x):
        # sum our entry with the result of our residual
        
        skip = x
        c1 = self.conv1(x)
        c2 = self.conv2(torch.cat([skip, c1], 1))
        c3 = self.conv3(torch.cat([skip, c1, c2], 1))
        c4 = self.conv4(torch.cat([skip, c1, c2, c3], 1))
        c5 = self.conv5(torch.cat([skip, c1, c2, c3, c4], 1))
        # scaling factor is currently 0.2
        return (c5 * 0.2) + skip
        #return x + self._residual(x)

class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResBlockDense(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    growth_channels=growth_channels,
                    kernel_size=kernel_size)
        self.rdb2 = ResBlockDense(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    growth_channels=growth_channels,
                    kernel_size=kernel_size)
        self.rdb3 = ResBlockDense(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    growth_channels=growth_channels,
                    kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)
        return (x * 0.2) + x # 0.2 is the scale factor of our dense blocks here



class SISR_Resblocks(nn.Module):
    def __init__(self, num_blocks):
        super(SISR_Resblocks, self).__init__()

        self.resblocks = []
        for i in range(num_blocks):
            self.resblocks.append(
                ResBlock(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3))
        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x):
        return self.resblocks(x)

class RRDB_Resblocks(nn.Module):
    def __init__(self, num_blocks):
        super(RRDB_Resblocks, self).__init__()

        self.resblocks = []
        for i in range(num_blocks):
            self.resblocks.append(
                RRDB(
                    in_channels=64,
                    out_channels=64,
                    growth_channels=32,
                    kernel_size=3)
                )
        self.resblocks = nn.Sequential(*self.resblocks)        

    def forward(self, x):
        return self.resblocks(x)


class Generator(nn.Module):

    def __init__(self, resblocks):
        super(Generator, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.PReLU,
            normalization=None)

        # ESRGAN poo-poos batch norm layers so we don't have one here compared to SRGAN

        # Our stack of rrdbs
        self.resblocks = resblocks

        # Why does the paper's source have a layer here post-resblock with no activation?
        self.post_res = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            normalization=None)

        # Upscaling layers
        self.upconv = Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.LeakyReLU,
            normalization=None)

        self.hrconv = Conv2d(
            in_channels=64,
            #in_channels=16,
            out_channels=64,
            #out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.LeakyReLU,
            normalization=None)

        #self.hrconv = RRDB(
        #    in_channels=16,
        #    out_channels=16,
        #    growth_channels=8,
        #    kernel_size=3
        #)

        self.rgbconv = Conv2d(
            in_channels=64,
            #in_channels=16,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.Tanh,
            normalization=None)

        # this is effectively our upscale factor
        #self.pixel_shuffle = nn.PixelShuffle(2)

        #self.prelu = nn.PReLU()

        #self.conv3 = Conv2d(
        #    in_channels=64,
        #    out_channels=3,
        #    kernel_size=9,
        #    stride=1,
        #    padding=9//2,
        #    activation=nn.Tanh,
        #    normalization=None)

    def forward(self, x):
        skip = self.conv1(x)
        # ignore the residual scaling paramter β for now
        # add the skip back at the end of our resblocks
        x = self.resblocks(skip) + skip
        x = self.post_res(x)
        #x = self.conv2(x)
        #x = self.pixel_shuffle(x)
        #x = self.prelu(x)
        #x = self.conv3(x)
        
        #x = self.pixel_shuffle(x)
        #x = self.prelu(x)
        x = self.upconv(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.hrconv(x)
        x = self.rgbconv(x)
        return x

## "borrowed" from https://github.com/xinntao/BasicSR/blob/master/basicsr/models/archs/discriminator_arch.py
class VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.
    It is used to train SRGAN and ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch=3, num_feat=64):
        super(VGGStyleDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, (
            f'Input spatial size must be 128x128, '
            f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


class VGG19FeatureNet(nn.Module):
    def __init__(self):
        super(VGG19FeatureNet, self).__init__()
        # Grab the 13th layer of the VGG19 net for feature detection
        self.truncated_model = nn.Sequential(*list(vgg19(pretrained = True).features))[:13]
        self.truncated_model.eval()

    def forward(self, x):
        return self.truncated_model(x)
>>>>>>> Stashed changes
