import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


# 自定义DropPath（避免依赖timm）
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# 自定义截断正态初始化
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(min=a, max=b)
        return tensor


class FeatureSelectionModule(nn.Module):
    """轻量级特征选择模块(FSM)，动态筛选重要特征"""

    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        self.dim = dim
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_attn(x)

        # 空间注意力
        spatial_weights = self.spatial_attn(x)

        # 组合注意力
        weighted_x = x * channel_weights * spatial_weights
        return weighted_x


class GatedCNNBlock(nn.Module):
    """MambaOut的核心门控卷积块"""

    def __init__(self, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()

        # 部分通道进行卷积
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)

        self.conv = nn.Conv2d(conv_channels, conv_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        # 分割特征
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)

        # 卷积路径
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # 门控融合
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut


class TinyViTBlock(nn.Module):
    """ViT的注意力块"""

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # 注意力层
        self.norm1 = nn.LayerNorm(dim)
        self.att = nn.MultiheadAttention(dim, num_heads, dropout=drop)

        # MLP层
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(drop)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 局部卷积增强
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        # 转换为序列
        x = x.view(B, H * W, C)

        # 注意力机制 - 关键修正：使用self.att而不是self.attn
        x = self.norm1(x)
        attn_out, _ = self.att(x, x, x)  # 使用self.att
        x = shortcut.view(B, H * W, C) + self.drop_path(attn_out)

        # 转换回空间格式
        x = x.view(B, H, W, C)

        # 局部卷积增强
        conv_in = x.permute(0, 3, 1, 2)
        conv_out = self.local_conv(conv_in)
        x = x + conv_out.permute(0, 2, 3, 1)

        # MLP
        mlp_in = x.view(B, H * W, C)
        mlp_in = self.norm2(mlp_in)
        mlp_out = self.mlp(mlp_in)
        x = x + self.drop_path(mlp_out).view(B, H, W, C)

        return x


class SoilNetHybrid(nn.Module):
    """融合MambaOut和TinyViT的土壤图像分类模型（加入特征选择模块）"""

    def __init__(self, in_chans=3, num_classes=5,
                 embed_dims=[64, 128],  # 减少参数
                 depths=[2, 2],  # 减少深度
                 num_heads=[2, 4],  # 注意力头数
                 conv_ratio=0.5,  # 部分通道卷积
                 window_size=7,  # 注意力窗口大小
                 head_dropout=0.3,  # 分类头dropout
                 use_fsm=True):  # 特征选择开关
        super().__init__()
        self.use_fsm = use_fsm

        # 共享特征提取层
        self.shared_stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.LayerNorm([64, 56, 56])
        )

        # 双分支特征提取
        self.local_branch = nn.ModuleList()
        self.global_branch = nn.ModuleList()

        # 特征选择模块
        if use_fsm:
            self.fsm_modules = nn.ModuleList()

        # 构建分支
        for i, dim in enumerate(embed_dims):
            # MambaOut分支 (局部特征)
            local_blocks = []
            for _ in range(depths[i]):
                local_blocks.append(GatedCNNBlock(
                    dim=dim,
                    conv_ratio=conv_ratio
                ))
            self.local_branch.append(nn.Sequential(*local_blocks))

            # ViT分支 (全局特征)
            global_blocks = []
            for _ in range(depths[i]):
                global_blocks.append(TinyViTBlock(
                    dim=dim,
                    num_heads=num_heads[i],
                    window_size=window_size
                ))
            self.global_branch.append(nn.Sequential(*global_blocks))

            # 为每层添加特征选择模块
            if use_fsm:
                self.fsm_modules.append(FeatureSelectionModule(dim))

        # 下采样层
        self.downsample = nn.ModuleList()
        for i in range(len(embed_dims) - 1):
            self.downsample.append(nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=2, padding=1),
                nn.LayerNorm([embed_dims[i + 1], 28 // (2 ** i), 28 // (2 ** i)])
            ))

        # 特征融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims[-1] * 2, embed_dims[-1] // 4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dims[-1] // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dims[-1] * 2),
            nn.Linear(embed_dims[-1] * 2, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes)
        )

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 共享特征提取
        x = self.shared_stem(x)  # [B, 64, 56, 56]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # 双分支处理
        for i, (local_block, global_block) in enumerate(zip(self.local_branch, self.global_branch)):
            # 局部分支处理
            local_feat = local_block(x)

            # 全局分支处理
            global_feat = global_block(x)

            # 应用特征选择模块
            if self.use_fsm:
                # 转换为通道优先格式
                local_c = local_feat.permute(0, 3, 1, 2)
                global_c = global_feat.permute(0, 3, 1, 2)

                # 特征筛选
                local_feat = self.fsm_modules[i](local_c).permute(0, 2, 3, 1)
                global_feat = self.fsm_modules[i](global_c).permute(0, 2, 3, 1)

            # 下采样 (除了最后一层)
            if i < len(self.downsample):
                # 合并特征用于下采样
                x = (local_feat + global_feat) / 2
                x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
                x = self.downsample[i](x)
                x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # 最终特征融合
        local_feat = local_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
        global_feat = global_feat.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 门控特征融合
        concat_feat = torch.cat([local_feat, global_feat], dim=1)
        gate_weights = self.fusion_gate(concat_feat)

        weighted_local = local_feat * gate_weights[:, 0:1]
        weighted_global = global_feat * gate_weights[:, 1:2]

        # 融合特征分类
        fused_feat = torch.cat([weighted_local, weighted_global], dim=1)
        fused_feat = F.adaptive_avg_pool2d(fused_feat, 1)  # 全局平均池化
        fused_feat = torch.flatten(fused_feat, 1)
        return self.head(fused_feat)


# 修改函数名称避免冲突
def build_soilnet_hybrid(num_classes=5, use_fsm=True, **kwargs):
    return SoilNetHybrid(
        num_classes=num_classes,
        embed_dims=[64, 128],
        depths=[2, 2],
        num_heads=[2, 4],
        conv_ratio=0.5,
        head_dropout=0.3,
        use_fsm=use_fsm,
        **kwargs
    )


# 测试代码保持隔离
if __name__ == "__main__":
    model = build_soilnet_hybrid(num_classes=5, use_fsm=True)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    print(f"输入尺寸: {input_tensor.shape} -> 输出尺寸: {output.shape}")