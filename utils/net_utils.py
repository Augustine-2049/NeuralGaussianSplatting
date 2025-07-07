import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureToRGBMLP(nn.Module):
    def __init__(self, in_features=64, hidden_features=128, out_features=3):
        super(FeatureToRGBMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )
        self._init_weights()

    def _init_weights(self):
        # 初始化权重
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [H, W, 64]
        H, W, C = x.shape
        x = x.view(-1, C)           # [H*W, 64]
        out = self.mlp(x)           # [H*W, 3]
        out = out.view(H, W, -1)    # [H, W, 3]
        return out

class DoubleConv(nn.Module):
    """(Conv2d => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=64, out_channels=3, base_channels=64):
        super(UNet, self).__init__()
        # Encoder   
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        # x: [H, W, 64] -> [1, 64, H, W]
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4 and x.shape[0] != 1:
            raise ValueError("Only single image input supported for this UNet3Layer")
        # Encoder
        e1 = self.enc1(x)  # [1, base, H, W]
        e2 = self.enc2(self.pool(e1))  # [1, base*2, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [1, base*4, H/4, W/4]
        # Decoder
        up2 = self.up2(e3)  # [1, base*2, H/2, W/2]
        cat2 = torch.cat([up2, e2], dim=1)  # [1, base*4, H/2, W/2]
        d2 = self.dec2(cat2)  # [1, base*2, H/2, W/2]
        up1 = self.up1(d2)  # [1, base, H, W]
        cat1 = torch.cat([up1, e1], dim=1)  # [1, base*2, H, W]
        d1 = self.dec1(cat1)  # [1, base, H, W]
        out = self.final_conv(d1)  # [1, 3, H, W]
        out = out.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return out

# 用法示例
def test_unet3layer():
    H, W = 64, 64
    x = torch.randn(H, W, 64)
    model = UNet()
    rgb = model(x)  # rgb.shape == [H, W, 3]
    print(rgb.shape)

class CNN(nn.Module):
    def __init__(self, in_channels=64, mid_channels=100, out_channels=81, kernel_size=5):
        super(CNN, self).__init__()
        padding = kernel_size // 2  # 保持空间尺寸不变
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [H, W, 64] -> [1, 64, H, W]
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4 and x.shape[0] != 1:
            raise ValueError("Only single image input supported for this CNN5Layer")
        x = self.relu(self.conv1(x))  # [1, 100, H, W]
        x = self.relu(self.conv2(x))  # [1, 100, H, W]
        x = self.relu(self.conv3(x))  # [1, 100, H, W]
        x = self.relu(self.conv4(x))  # [1, 100, H, W]
        x = self.conv5(x)             # [1, 81, H, W]
        out = x.squeeze(0).permute(1, 2, 0)  # [H, W, 81]
        return out

class Denoiser(nn.Module):
    def __init__(self, kernel_size=9):
        super(Denoiser, self).__init__()
        self.kernel_size = kernel_size  
        self.padding = kernel_size // 2

    def forward(self, unet_out, cnn_out):
        """
        unet_out: [H, W, 3]
        cnn_out: [H, W, 81]  # 81 = 9*9
        """
        H, W, C = unet_out.shape
        assert C == 3
        # 1. 生成动态卷积核
        kernels = cnn_out.view(H, W, self.kernel_size, self.kernel_size)  # [H, W, 9, 9]
        # 2. 对unet_out每个像素做动态卷积
        # 先pad
        unet_out = unet_out.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        unet_pad = F.pad(unet_out, [self.padding]*4, mode='reflect')  # [1, 3, H+8, W+8]
        # unfold提取每个像素的9x9邻域
        patches = F.unfold(unet_pad, kernel_size=self.kernel_size)  # [1, 3*81, H*W]
        patches = patches.view(1, 3, self.kernel_size*self.kernel_size, H, W)  # [1, 3, 81, H, W]
        # 动态核加权
        kernels = kernels.permute(2, 3, 0, 1).unsqueeze(0)  # [1, 9, 9, H, W]
        kernels = kernels.reshape(1, 1, self.kernel_size*self.kernel_size, H, W)  # [1, 1, 81, H, W]
        out = (patches * kernels).sum(dim=2)  # [1, 3, H, W]
        out = out.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return out


# 用法示例
def test_cnn5layer():
    H, W = 32, 32
    x = torch.randn(H, W, 64)
    model = CNN()
    y = model(x)  # y.shape == [H, W, 81]
    print(y.shape)

if __name__ == "__main__":
    test_unet3layer()
    test_cnn5layer()