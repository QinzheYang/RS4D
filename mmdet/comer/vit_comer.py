from mmcv.cnn.bricks import DropPath
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from torch.nn.init import normal_

class TransformerBlock(nn.Module):
    """
    Transformer Block.
    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        qkv_bias (bool): Whether to add bias to qkv projections.
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float): DropPath rate.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForwardNetwork(dim, int(dim * mlp_ratio), act_layer, drop)

    def forward(self, x, H, W):
        # Self-Attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # Feed-Forward Network
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to add bias to qkv projections.
        attn_drop (float): Attention dropout rate.
        proj_drop (float): Output dropout rate.
    """
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network.
    Args:
        dim (int): Embedding dimension.
        hidden_dim (int): Hidden dimension.
        act_layer (nn.Module): Activation layer.
        drop (float): Dropout rate.
    """
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        norm_layer (nn.Module): Normalization layer.
        flatten (bool): Whether to flatten the output.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, grid_size, grid_size]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size

@MODELS.register_module()
class ViTCoMer(nn.Module):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_CTI=True, pretrained=None, with_cp=False,
                 use_CTI_toV=True, use_CTI_toC=True, cnn_feature_interaction=True, dim_ratio=6.0, *args, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.use_CTI_toC = use_CTI_toC
        self.use_CTI_toV = use_CTI_toV
        self.add_vit_feature = add_vit_feature
        self.embed_dim = 768  # Default embed_dim for ViT

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=pretrain_size,
            patch_size=16,
            in_chans=3,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm,
            flatten=True
        )
        num_patches = self.patch_embed.num_patches

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))  # +1 for cls_token
        self.pos_drop = nn.Dropout(p=0.1)  # Default dropout rate

        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, 0.1, len(interaction_indexes))]  # DropPath rates
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(len(interaction_indexes))
        ])

        # Other components
        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = CNN(inplanes=conv_inplane, embed_dim=self.embed_dim)
        self.interactions = nn.Sequential(*[
            CTIBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                     init_values=init_values, drop_path=0.1,  # Default drop_path_rate
                     norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                     cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                     use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                     use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                     dim_ratio=dim_ratio,
                     cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else
                     cnn_feature_interaction[i],
                     extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        # Initialize weights
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        #self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    #def _init_deform_weights(self, m):
        #if isinstance(m, MSDeformAttn):
            #m.init_weights()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        print("x0",x.shape)
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        print("c0", c1.shape,c2.shape,c3.shape,c4.shape)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        print("c1",c2.shape, c3.shape, c4.shape)
        c = torch.cat([c2, c3, c4], dim=1)
        print("c2", c.shape)
        # Patch Embedding forward
        print("x1", x.shape)
        x, H, W = self.patch_embed(x)#emb=768,patch=24x24x3=576
        print("x2", x.shape)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        print("x3", x.shape)
        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            print("x",x.shape)
            print("c",c.shape)
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                     (h // 16, w // 16),
                                     (h // 32, w // 32)],
                                    dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                            (h // 16, w // 16),
                                            (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        print("342",x.shape)
        if N==3024:
            H=48
            W=63
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CTIBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25,
                 init_values=0., deform_ratio=1.0, extra_CTI=False, with_cp=False,
                 use_CTI_toV=True, use_CTI_toC=True, dim_ratio=6.0, cnn_feature_interaction=False):
        super().__init__()
        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC

        # === CTI-to-Vision（ViT特征增强CNN特征） ===
        if use_CTI_toV:
            self.cti_tov = MSDeformAttn(
                embed_dims=dim,
                num_heads=num_heads,
                num_points=n_points,
                num_levels=3,  # 对应3个尺度
                value_proj_ratio=deform_ratio
            )
            self.gamma = nn.Parameter(init_values * torch.ones(dim))
            self.ffn = ConvFFN(dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # === CTI-to-CNN（CNN特征增强ViT特征） ===
        if use_CTI_toC:
            self.cti_toc = MSDeformAttn(
                embed_dims=dim,
                num_heads=num_heads,
                num_points=n_points,
                num_levels=1,  # 单尺度
                value_proj_ratio=deform_ratio
            )
            self.cnn_feature_interaction = cnn_feature_interaction
            if cnn_feature_interaction:
                self.cfinter = MSDeformAttn(
                    embed_dims=dim,
                    num_heads=num_heads,
                    num_points=n_points,
                    num_levels=3,
                    value_proj_ratio=deform_ratio
                )

        # === 其他模块 ===
        self.mrfp = ConvFFN(dim, hidden_features=int(dim * dim_ratio))
        self.norm = norm_layer(dim)

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        B, N, C = x.shape

        # === 1. 多分辨率特征传播 (MRFP) ===
        c = self.mrfp(c, H, W)  # [B, 3024, C]

        # === 2. 拆分多尺度特征 ===
        # 假设c由三个尺度拼接: 48x48 (2304), 24x24 (576), 12x12 (144) → 总长度3024
        c1 = c[:, :2304, :]  # 尺度1: 48x48
        c2 = c[:, 2304:2880, :]  # 尺度2: 24x24
        c3 = c[:, 2880:, :]  # 尺度3: 12x12

        # === 3. CTI-to-Vision: 用ViT特征增强CNN特征 ===
        if self.use_CTI_toV:
            # 生成多尺度参考点
            ref_points = self._generate_multi_scale_refpoints(H, W, x.device)  # [B, N, 3, 2]

            # 执行可变形注意力
            c = self.cti_tov(
                query=x,  # [B, 576, C]
                reference_points=ref_points,  # [B, 576, 3, 2]
                value=torch.cat([c1, c2, c3], dim=1),  # [B, 3024, C]
                spatial_shapes=torch.tensor([[48, 48], [24, 24], [12, 12]], device=x.device),
                level_start_index=torch.tensor([0, 2304, 2880], device=x.device)
            )
            x = x + self.gamma * c

        # === 4. 通过Transformer Blocks ===
        for blk in blocks:
            x = blk(x, H, W)

        # === 5. CTI-to-CNN: 用CNN特征增强ViT特征 ===
        if self.use_CTI_toC:
            # 生成单尺度参考点（对应ViT的24x24特征）
            ref_points = self._generate_single_scale_refpoints(H, W, x.device)  # [B, 576, 1, 2]

            # 执行可变形注意力
            c = self.cti_toc(
                query=c2,  # 选择中间尺度（24x24）
                reference_points=ref_points,
                value=x,
                spatial_shapes=torch.tensor([[24, 24]], device=x.device),
                level_start_index=torch.tensor([0], device=x.device)
            )
            c2 = c2 + c

        return x, torch.cat([c1, c2, c3], dim=1)

    def _generate_multi_scale_refpoints(self, H, W, device):
        """生成多尺度参考点坐标（归一化到[0,1]）"""
        ref_points = []
        # 尺度1: 48x48
        y48, x48 = torch.meshgrid(torch.linspace(0, 1, 48), torch.linspace(0, 1, 48))
        ref48 = torch.stack((x48, y48), dim=-1).view(1, -1, 2).to(device)  # [1, 2304, 2]

        # 尺度2: 24x24
        y24, x24 = torch.meshgrid(torch.linspace(0, 1, 24), torch.linspace(0, 1, 24))
        ref24 = torch.stack((x24, y24), dim=-1).view(1, -1, 2).to(device)  # [1, 576, 2]

        # 尺度3: 12x12
        y12, x12 = torch.meshgrid(torch.linspace(0, 1, 12), torch.linspace(0, 1, 12))
        ref12 = torch.stack((x12, y12), dim=-1).view(1, -1, 2).to(device)  # [1, 144, 2]

        # 合并并重复Batch维度
        ref_points = torch.cat([ref48, ref24, ref12], dim=1)  # [1, 3024, 2]
        ref_points = ref_points.repeat(8, 1, 1)  # [B, 3024, 2]
        return ref_points.unsqueeze(2)  # [B, 3024, 3, 2]

    def _generate_single_scale_refpoints(self, H, W, device):
        """生成单尺度参考点坐标（归一化到[0,1]）"""
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
        ref = torch.stack((x, y), dim=-1).view(1, -1, 2).to(device)  # [1, H*W, 2]
        return ref.repeat(x.size(0), 1, 1).unsqueeze(2)  # [B, H*W, 1, 2]

class CTIBlock1(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=nn.LayerNorm, drop=0., drop_path=0.,
                 with_cffn=True, cffn_ratio=0.25, init_values=0., deform_ratio=1.0, extra_CTI=False,
                 with_cp=False, use_CTI_toV=True, use_CTI_toC=True, dim_ratio=6.0, cnn_feature_interaction=False):
        super().__init__()
        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC
        self.with_cp = with_cp

        # CTI-to-Vision (ViT to CNN)
        if use_CTI_toV:
            self.cti_tov = MSDeformAttn(
                embed_dims=dim,
                num_heads=num_heads,
                num_points=n_points,
                num_levels=3,
                value_proj_ratio=deform_ratio
            )
            self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.ffn = ConvFFN(dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # CTI-to-CNN (CNN to ViT)
        if use_CTI_toC:
            self.cti_toc = MSDeformAttn(
                embed_dims=dim,
                num_heads=num_heads,
                num_points=n_points,
                num_levels=1,
                value_proj_ratio=deform_ratio
            )
            self.cnn_feature_interaction = cnn_feature_interaction
            if cnn_feature_interaction:
                self.cfinter = MSDeformAttn(
                    embed_dims=dim,
                    num_heads=num_heads,
                    num_points=n_points,
                    num_levels=3,
                    value_proj_ratio=deform_ratio
                )

        # Extra CTI blocks
        if extra_CTI:
            self.extra_CTIs = nn.ModuleList([
                MSDeformAttn(
                    embed_dims=dim,
                    num_heads=num_heads,
                    num_points=n_points,
                    num_levels=1,
                    value_proj_ratio=deform_ratio
                )
                for _ in range(4)
            ])
        else:
            self.extra_CTIs = None

        # Multi-Resolution Feature Propagation
        self.mrfp = ConvFFN(dim, hidden_features=int(dim * dim_ratio))

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        B, N, C = x.shape

        # Generate deformable inputs
        deform_inputs = deform_inputs_only_one(x, H * 16, W * 16)

        # CTI-to-Vision (ViT to CNN)
        if self.use_CTI_toV:
            # Multi-Resolution Feature Propagation
            print("423", c.shape, H, W)
            c = self.mrfp(c, H, W)
            print("425", c.shape, H, W)
            c_select1, c_select2, c_select3 = c[:, :H * W * 4, :], c[:, H * W * 4:H * W * 4 + H * W, :], c[:, H * W * 4 + H * W:, :]
            c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)
            print("427", c.shape, H, W)
            # Apply CTI-to-Vision
            x = x + self.gamma * self.cti_tov(x, deform_inputs[0], c, deform_inputs[1], deform_inputs[2])
            x = x + self.drop_path(self.ffn(self.ffn_norm(x), H, W))

        # Pass through transformer blocks
        for blk in blocks:
            x = blk(x, H, W)

        # CTI-to-CNN (CNN to ViT)
        if self.use_CTI_toC:
            c = c + self.cti_toc(c, deform_inputs2[0], x, deform_inputs2[1], deform_inputs2[2])
            if self.cnn_feature_interaction:
                c = c + self.cfinter(c, deform_inputs[0], x, deform_inputs[1], deform_inputs[2])

        # Extra CTI blocks
        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = c + cti(c, deform_inputs2[0], x, deform_inputs2[1], deform_inputs2[2])

        return x, c

class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4

def deform_inputs_only_one(x, H, W):
    """
    生成多尺度可变形注意力所需的输入。
    Args:
        x (torch.Tensor): 输入特征图，形状为 [B, C, H, W]。
        H (int): 特征图的高度。
        W (int): 特征图的宽度。
    Returns:
        deform_inputs (list): 包含参考点、空间形状和层级索引的列表。
    """
    device = x.device
    spatial_shapes = torch.as_tensor([(H // 8, W // 8),
                                     (H // 16, W // 16),
                                     (H // 32, W // 32)],
                                    dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(H // 8, W // 8),
                                            (H // 16, W // 16),
                                            (H // 32, W // 32)], device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs


class MSDeformAttn(nn.Module):
    def __init__(self, embed_dims, num_heads, num_points, num_levels, value_proj_ratio=1.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.value_proj_ratio = value_proj_ratio

        # Projection layers
        self.value_proj = nn.Linear(embed_dims, int(embed_dims * value_proj_ratio))
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.output_proj = nn.Linear(int(embed_dims * value_proj_ratio), embed_dims)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        B, Len_q, C = query.shape
        num_levels = spatial_shapes.shape[0]

        # === 1. Value投影和分头 ===
        value = self.value_proj(value)
        value = value.view(B, -1, self.num_heads, C // self.num_heads)

        # === 2. 生成采样偏移量 ===
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            B, Len_q, self.num_heads, num_levels, self.num_points, 2
        )

        # === 3. 生成注意力权重 ===
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            B, Len_q, self.num_heads, num_levels * self.num_points
        ).softmax(dim=-1)
        attention_weights = attention_weights.view(
            B, Len_q, self.num_heads, num_levels, self.num_points
        )

        print("value", value.shape)  # torch.Size([8, 3024, 6, 128])
        print("sampling_offsets", sampling_offsets.shape)  # torch.Size([8, 576, 6, 3, 4, 2])
        print("attention_weights", attention_weights.shape)  # torch.Size([8, 576, 6, 3, 4])

        # === 4. 采样点的坐标计算 ===
        # reference_points 的形状是 [8, 3024, 1, 2]，需要插值到 [8, 576, 1, 2]
        reference_points = reference_points.view(B, -1, 1, 2)  # [8, 3024, 1, 2]
        print(reference_points.shape)
        part1 = reference_points[:, :2304, :, :]  # 前 48x48 => 2304
        part2 = reference_points[:, 2304:2880, :, :]  # 中间 24x24 => 576
        part3 = reference_points[:, 2880:, :, :]  # 后 12x12 => 144

        # 2. 平均前 48x48 成 576
        # 48x48 -> 576 (通过平均)
        part1 = part1.view(B, 48, 48, 1, 2).mean(dim=(1, 2)).view(B, 1, 1, 2)  # 形状为 [8, 1, 1, 2]
        part1 = part1.expand(-1, 576, -1, -1)  # 扩展到 [8, 576, 1, 2]

        # 3. 中间 24x24 不变
        part2 = part2.view(B, 24 * 24, 1, 2)  # 变为 [8, 576, 1, 2]，原本是 [8, 576, 1, 2]

        # 4. 后 12x12 插值成 576
        part3 = part3.view(B, 12,12, 1, 2)  # 变为 [8, 12, 12, 1, 2]
        part3 = part3.view(B, 12, 12, 2)  # 将最后两个维度合并为通道维度
        part3 = part3.permute(0, 3, 1, 2)  # 调整为 [B, C, H, W]，其中 C=2, H=12, W=12

        # 插值到 [B, 2, 24, 24]
        part3 = F.interpolate(part3, size=(24, 24), mode='bilinear', align_corners=False)

        # 调整形状为 [B, 576, 1, 2]
        part3 = part3.view(B, 2, -1)  # 将空间维度展平为 [B, 2, 576]
        part3 = part3.permute(0, 2, 1)  # 调整为 [B, 576, 2]
        part3 = part3.view(B, 576, 1, 2)  # 最终形状

        # 5. 平均三者
        final_output = (part1 + part2 + part3) / 3

        reference_points = final_output[:, :, None, :, None, :].expand(
            B, Len_q, self.num_heads, num_levels, 1, 2
        )  # [8, 576, 6, 3, 1, 2]
        print(reference_points.shape,"reference_points")

        # 将采样偏移量与参考点相加，得到最终采样点坐标
        sampling_coords = sampling_offsets + reference_points
        sampling_coords = sampling_coords.view(B, Len_q, self.num_heads, num_levels * self.num_points, 2)

        # === 5. 多尺度采样和加权求和 ===
        # 初始化输出张量
        output = torch.zeros((B, Len_q, self.num_heads, C // self.num_heads), device=query.device)

        print("output.shape",output.shape)
        print("sampling_coords", sampling_coords.shape)
        print("attention_weights", attention_weights.shape)
        for level in range(num_levels):
            # 当前层级的起始索引
            start_idx = level_start_index[level]
            end_idx = start_idx + spatial_shapes[level][0] * spatial_shapes[level][1]

            # 当前层级的采样点坐标
            level_sampling_coords = sampling_coords[:, :, :, level * self.num_points:(level + 1) * self.num_points, :]

            # 将采样点坐标映射到特征图的索引
            sampling_x = torch.floor(level_sampling_coords[..., 0]).long()
            sampling_y = torch.floor(level_sampling_coords[..., 1]).long()

            # 对每个采样点进行采样
            for i in range(self.num_points):
                # 当前采样点的坐标
                sampling_x_i = sampling_x[:, :, :, i]
                sampling_y_i = sampling_y[:, :, :, i]

                # 采样点的权重
                weights = attention_weights[:, :, :, level, i]

                # 对每个采样点进行采样
                sampled_value = value[:, start_idx:end_idx, :, :].view(
                    B, spatial_shapes[level][0], spatial_shapes[level][1], self.num_heads, -1
                )
                sampled_value = sampled_value[
                    sampling_x_i,
                    sampling_y_i,
                    :
                ]
                print("weights",weights.shape)
                print("sampled_value",sampled_value.shape)
                # 加权求和
                tmp=weights[:, :, :, None] * sampled_value
                print("tmp",tmp.shape)
                print("outputd",output.shape)
                output = output+tmp

        # 将输出张量的形状调整为 [B, Len_q, C]
        output = output.view(B, Len_q, -1)

        # === 6. 输出投影 ===
        output = self.output_proj(output)  # [B, Len_q, C]
        return output