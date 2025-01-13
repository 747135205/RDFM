import re

import torch
from efficientnet_pytorch import EfficientNet
from einops import rearrange
from torch import einsum
from torch import nn
import torch.nn.functional as F
import math

class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int),indexing='ij')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number),indexing='ij')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),indexing='ij')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),
        # stride=(num_param,1,1),bias= False) using 1 × 1 Conv x_offset = x_offset.permute(0,1,4,2,3), then,
        # x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1),
        # bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result





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
        return x

class AK_STVit_CBAM_row(nn.Module):

    def __init__(self, dim):
        super().__init__()
        # self.AK=AKConv(inc=dim,outc=dim,num_param=3)
        self.STVit=StokenAttention(dim=dim,stoken_size=[8,8])
        self.CBAM=CBAM(in_planes=dim)
        self.conv=nn.Conv2d(2*dim,dim,kernel_size=1)
    def forward(self, x):
        res=x
        # x1=self.AK(x)
        x2=self.STVit(x)
        x3=self.CBAM(x)
        x = torch.cat((x2, x3), dim=1)
        x=self.conv(x)
        return x+res


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
        self.CBAM=CBAM(in_planes=dim)
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
        x=self.CBAM(x)
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
