import re

import torch
from efficientnet_pytorch import EfficientNet
from einops import rearrange
from torch import einsum
from torch import nn
import torch.nn.functional as F
import math

from torch.nn import init

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        avg_result = self.avgpool(x)  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        max_out = self.se(max_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        avg_out = self.se(avg_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        output = self.sigmoid(max_out + avg_out)  # 相加,然后通过sigmoid获得权重:(B,C,1,1)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result = torch.cat([max_result, avg_result], 1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        output = self.conv(result)  # 然后重新降维为1维:(B,1,H,W)
        output = self.sigmoid(output)  # 通过sigmoid获得权重:(B,1,H,W)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ChannelAttention = ChannelAttention(channel=channel, reduction=reduction)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        residual = x
        out = x * self.ChannelAttention(x)  # 将输入与通道注意力权重相乘: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)
        out = out * self.SpatialAttention(out)  # 将更新后的输入与空间注意力权重相乘:(B,C,H,W) * (B,1,H,W) = (B,C,H,W)
        return out + residual


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention_2(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape  #x: 输入张量，形状为 (batch_size, channels, height, width)。
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)
        #将输入映射为 Query、Key、Value，然后按头数拆分。

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        #stoken_size:结构化标记的尺寸。 n_iter结构化标记的迭代次数。qk_scale缩放因子。dim输入向量维度
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention_2(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        #对张量进行填充，保证他可以被h与w整除
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        # dim 输入向量维度 heads:多头自注意力维数 dropout的概率 dim_head 头注意力的维度
        super().__init__()
        inner_dim = dim_head * heads  # 中间层维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要投影输出，如果多头为 1 且维度相同，则无需投影。

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子

        self.attend = nn.Softmax(dim=-1)  # 计算注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  # x 形状为 (batch_size, sequence_length, input_dim)。
        b, n, _, h = *x.shape, self.heads  # 从输入张量中获取 batch_size、sequence_length、和 heads 数量。
        #torch.Size([18, 2, 1024])
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 将输入映射为 Query、Key、Value，并按最后一个维度拆分成三个部分。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # 使用 rearrange 函数将每个部分重排，形成多头机制。

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # 计算点积，然后通过 self.scale 进行缩放。
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 使用注意力权重对 Value 进行加权求和，最后重排得到多头自注意力的输出。
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        # dim: 输入向量的维度。depth: Transformer 模块中包含的层数。heads: 多头自注意力的头数。dim_head: 每个注意力头的维度。mlp_dim: FeedForward 层中隐藏层的维度。dropout: 模型中的 dropout 概率
        super().__init__()
        self.CBAM=CBAMBlock(channel=1024,reduction=16,kernel_size=7)
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 利用for循环完成attention网络结构架构，前馈神经网络结构的 FeedForward 层。
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=0))
            ]))

    def forward(self, x): #这里的前向函数只做残差工作，向量的嵌入工作已经提前完成了
        for attn, ff in self.layers:  # 层与层之间通过残差连接进行连接，这有助于信息的流动和梯度的传播
            x = attn(x) + x
            x = ff(x) + x
        res = x
        x = to_4d(x, 2, 2)
        x = self.CBAM(x)
        x = to_3d(x) + res
        return x


class EfficientViT(nn.Module):
    def __init__(self, config, channels=512, selected_efficient_net=0):
        super().__init__()

        image_size = config['model']['image-size']
        patch_size = config['model']['patch-size']
        num_classes = config['model']['num-classes']
        dim = config['model']['dim']
        depth = config['model']['depth']
        heads = config['model']['heads']
        mlp_dim = config['model']['mlp-dim']
        emb_dim = config['model']['emb-dim']
        dim_head = config['model']['dim-head']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.selected_efficient_net = selected_efficient_net

        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
            checkpoint = torch.load("weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
                                    map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()},
                                               strict=False)

        for i in range(0, len(self.efficient_net._blocks)):
            for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                if i >= len(self.efficient_net._blocks) - 3:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        num_patches = (7 // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(emb_dim, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.sovit= StokenAttention(dim=1280,stoken_size=[8,8])
        # dim 输入向量维度 heads:多头自注意力维数 dropout的概率 dim_head 头注意力的维度. 剩depth和mlp_dim
        # dim,depth不需要（构造attention网络深度的指标），heads默认参数，

        self.to_cls_token = nn.Identity() #恒等映射，提取cls字段

        # 分类头部分，完成将结果最后重新映射到num_classes的范围之中
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.efficient_net.extract_features(img)  # 提取图像特征，输出大小为1280x7x7。1280代表通道数，7*7代表卷积特征方块大小，即一个patch
        x2=self.sovit(x)
        #torch.Size([18, 2560, 7, 7])
        # x = torch.cat((x, x2), dim=1) #修改2，未完工
        #torch.Size([18, 1280, 7, 7])
        # x = self.features(img)
        '''
        for im in img:
            image = im.cpu().detach().numpy()
            image = np.transpose(image, (1,2,0))
            cv2.imwrite("images/image"+str(randint(0,1000))+".png", image)
        
        x_scaled = []
        for idx, im in enumerate(x):
            im = im.cpu().detach().numpy()
            for patch_idx, patch in enumerate(im):
                patch = (255*(patch - np.min(patch))/np.ptp(patch)) 
                im[patch_idx] = patch
                #cv2.imwrite("patches/patches_"+str(idx)+"_"+str(patch_idx)+".png", patch)
            x_scaled.append(im)
        x = torch.tensor(x_scaled).cuda()   
        '''

        # x2 = self.features(img)
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        #torch.Size([18, 1, 125440])
        #将特征张量 x 重排为二维张量 y，其中每个 patch 被视为一个样本。
        # y2 = rearrange(x2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        y2 = rearrange(x2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)#修改的版本一
        y = self.patch_to_embedding(y) # 将每个 patch 映射为嵌入向量 y就是定义中的patch_dim的大小
        y2 = self.patch_to_embedding(y2)#修改的版本一
        #torch.Size([18, 1, 1024])
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) #将分类标记和嵌入向量拼接在一起。
        cls_tokens2 = self.cls_token.expand(x2.shape[0], -1, -1)#修改的版本一
        #torch.Size([18, 1, 1024])
        x = torch.cat((cls_tokens, y), 1) #对位置嵌入进行处理并添加到输入中。
        x2= torch.cat((cls_tokens2, y2), 1)#修改的版本一
        #torch.Size([18, 2, 1024])
        #torch.Size([18, 2, 1024]) batch_size、sequence_length
        # print(x.shape)
        x= torch.cat((x,x2),1)#修改的版本一
        shape = x.shape[0]
        # 下列代码即完成了对VIT过程的调用，所谓的VIT就在于对输入量预处理之上
        x += self.pos_embedding[0:shape]
        x = self.dropout(x)
        #torch.Size([18, 2, 1024])
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])

        return self.mlp_head(x)
