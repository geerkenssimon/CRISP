import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class CustomConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mode='conv', double=True):
        super().__init__()

        if mode == 'conv':
            blocks = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            if double:
                blocks += [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ]
        elif mode == 'ds':
            blocks = [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False,
                          stride=stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            if double:
                blocks += [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False,
                              stride=stride),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ]
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mode='conv', double=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            CustomConv(in_channels, out_channels, mode=mode, double=double)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, mode='conv', double=True, merge=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = CustomConv(in_channels, out_channels, mode=mode, double=double)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if self.conv.conv[0]._get_name() != 'Identity':
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        else:
            return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, wm=0.5, conv_mode='conv', double=True,
                 ltd=1, output_size=[512, 1024]):
        super(UNet, self).__init__()
        if output_size is None:
            output_size = [512, 1024]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.output_size = output_size
        self.f1 = int(round(wm * 64))
        self.f2 = int(round(wm * 128))
        self.f3 = int(round(wm * 256))
        self.f4 = int(round(wm * 512))
        self.f5 = int(round(wm * 1024))

        self.ltd = ltd

        if ltd > 0:
            ds_blocks = [
                nn.Conv2d(3, self.f1, kernel_size=3, stride=2),
                nn.BatchNorm2d(self.f1),

            ]
            n_channels = self.f1
            for i in range(1, ltd):
                ds_blocks.append(
                    CustomConv(self.f1, self.f1, mode=conv_mode, double=False, stride=2)
                )
            self.down = nn.Sequential(*ds_blocks)
            follow_up_mode = 'ds'
        else:
            follow_up_mode = 'conv'

        self.inc = CustomConv(n_channels, self.f1, mode=follow_up_mode, double=double)
        self.down1 = Down(self.f1, self.f2, mode=conv_mode, double=double)
        self.down2 = Down(self.f2, self.f3, mode=conv_mode, double=double)
        self.down3 = Down(self.f3, self.f4, mode=conv_mode, double=double)
        self.down4 = Down(self.f4, self.f4, mode=conv_mode, double=double)
        self.up1 = Up(self.f5, self.f3, bilinear, mode=conv_mode, double=double)
        self.up2 = Up(self.f4, self.f2, bilinear, mode=conv_mode, double=double)
        self.up3 = Up(self.f3, self.f1, bilinear, mode=conv_mode, double=double)
        self.up4 = Up(self.f2, self.f1, bilinear, mode=conv_mode, double=double)
        self.up = nn.Upsample(size=tuple(self.output_size), mode='bilinear')
        self.outc = OutConv(self.f1, n_classes)

    def forward(self, x):
        if self.ltd > 0:
            y = self.down(x)
            x1 = self.inc(y)
        else:
            x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up(x)
        logits = self.outc(x)
        
        return logits
