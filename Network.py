import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """MobileNet核心模块：深度可分离卷积（Depthwise + Pointwise）"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 深度卷积（Depthwise）：每个通道单独卷积，groups=in_channels 把输入通道分成 N 个独立小组，每组单独卷积；普通卷积（groups=1）
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 逐点卷积（Pointwise）：1x1卷积融合通道信息
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    """MobileNetV1主干网络"""
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super().__init__()
        self.alpha = width_multiplier  # 宽度乘数α，控制通道数缩放 是 MobileNet 用来灵活定制轻量化模型的关键开关

        # 初始标准卷积层（论文中第一个层用标准卷积）
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, int(32 * self.alpha), kernel_size=3, stride=2, 
                      padding=1, bias=False),
            nn.BatchNorm2d(int(32 * self.alpha)),
            nn.ReLU(inplace=True)
        )

        # 深度可分离卷积堆叠（对齐论文Table 1的结构）
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(int(32*self.alpha), int(64*self.alpha), stride=1),
            DepthwiseSeparableConv(int(64*self.alpha), int(128*self.alpha), stride=2),
            DepthwiseSeparableConv(int(128*self.alpha), int(128*self.alpha), stride=1),
            DepthwiseSeparableConv(int(128*self.alpha), int(256*self.alpha), stride=2),
            DepthwiseSeparableConv(int(256*self.alpha), int(256*self.alpha), stride=1),
            DepthwiseSeparableConv(int(256*self.alpha), int(512*self.alpha), stride=2),
            # 连续5个stride=1的512→512块
            *[DepthwiseSeparableConv(int(512*self.alpha), int(512*self.alpha), stride=1) 
              for _ in range(5)],
            DepthwiseSeparableConv(int(512*self.alpha), int(1024*self.alpha), stride=2),
            DepthwiseSeparableConv(int(1024*self.alpha), int(1024*self.alpha), stride=1)
        )

        # 分类头：平均池化 + 全连接
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * self.alpha), num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 测试模型前向传播
if __name__ == "__main__":
    # 创建模型（默认ImageNet 1000类，α=1.0）
    model = MobileNetV1(num_classes=1000, width_multiplier=1.0)
    
    # 随机输入：batch_size=2, 3通道, 224×224
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"输入形状: {x.shape}")      # torch.Size([2, 3, 224, 224])
    print(f"输出形状: {output.shape}")  # torch.Size([2, 1000])