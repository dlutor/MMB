# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import TransformerEncoder
from .Encoder import DropPath, Mlp, partial, _init_weights, Block, _get_clones, Attention


def torch_norm(x, dim=-1) -> torch.tensor:
    return torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)


class MLPFusion(nn.Module):
    def __init__(self,
                 dim,   cross_num=1,# 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.cross_num = cross_num
        self.mlp = _get_clones(nn.Linear(dim, cross_num * dim, bias=qkv_bias), N=cross_num)
        self.proj = _get_clones(nn.Linear(dim, dim), N=cross_num)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) and not (isinstance(xs, torch.Tensor) and len(xs.size()) == 3):
            xs = [xs]
        B, C = xs[0].size()
        xs_, xs__, x_list = [], [], []
        for i in range(self.cross_num):
            x_ = self.mlp[i](xs[i]).view(B, self.cross_num, C).permute(1, 0, 2)
            xs_.append(x_)
        for i in range(self.cross_num):
            x__ = torch.cat([xs_[j][i].unsqueeze(0) for j in range(self.cross_num)], dim=0)
            xs__.append(x__)

        for i in range(self.cross_num):
            x = xs__[i].sum(0).view(B, C)
            x = self.proj_drop(self.proj[i](x))
            x_list.append(x)
        return x_list



class AttentionBlock(nn.Module):
    def __init__(self,
                 dim, cross_num,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attention: nn.Module=MLPFusion,):
        super().__init__()
        self.norm1 = _get_clones(norm_layer(dim), N=cross_num)
        self.attn = attention(dim, cross_num, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = _get_clones(norm_layer(dim), N=cross_num)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _get_clones(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio), N=cross_num)


    def forward(self, x):
        norm_xs, x_list = [], []
        for i, xi in enumerate(x):
            norm_x = self.norm1[i](xi)
            norm_xs.append(norm_x)
        xs = self.attn(norm_xs)
        for i, xi in enumerate(xs):
            # attn = xi
            x_ = x[i] + self.drop_path(xi)
            # x_ = x[i] + self.drop_path((self.gate[i](x[i] + attn) + attn )* x[i])
            x_ = x_ + self.drop_path(self.mlp[i](self.norm2[i](x_)))

            x_list.append(x_)
        return x_list


class AttentionEncoder(nn.Module):
    def __init__(self, cross_num=3, embed_dim=300, depth=8, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None, attention: nn.Module=MLPFusion):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.norm = _get_clones(norm_layer(embed_dim), N=cross_num)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=embed_dim, cross_num=cross_num, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                                norm_layer=norm_layer, act_layer=act_layer, attention=attention)
            for i in range(depth)
        ])

        self.apply(_init_weights)

    def forward(self, *x): # [B, L, F]
        norm_xs, x_list = [], []
        for xi in x:
            xi = self.pos_drop(xi)
            norm_xs.append(xi)
        x = self.blocks(x)
        for i, xi in enumerate(x):
            xi = self.norm[i](xi)
            x_list.append(xi)
        return x_list



class BaseAttentionModel(nn.Module):
    def __init__(self,
                 num_cls=6,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dims=(300, 35, 74),
                 d_model=512,
                 encoder_dims=None,
                 t_length=60,
                 nhead=4,
                 encoder_heads=None,
                 dim_ff_ratio=4.0,
                 dropout=0.1,
                 activation=None, attention=None, fusion:callable=sum, aligned=False, encoder=None, mask=None):
        super().__init__()
        self.cross_num = len(dims)
        self.aligned = aligned
        self.mask = mask
        if encoder_dims is None:
            embedding_dims = [d_model] * self.cross_num
            encoder_head_dims = [0] * self.cross_num
        else:
            embedding_dims = encoder_dims
            encoder_head_dims = [d_model] * self.cross_num
        if encoder_heads is None:
            encoder_heads = [nhead] * self.cross_num
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, num_cls)
        )
        self.heads = _get_clones(self.head, N=self.cross_num)
        self.modal_cls = nn.Linear(d_model, self.cross_num + 1)
        self.src_linears = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], embedding_dims[i]),
                                         nn.LayerNorm(embedding_dims[i]),
                                         nn.Dropout(0.1)) if embedding_dims[i] > 0 else nn.Identity() for i in range(self.cross_num)])
        if isinstance(num_encoder_layers,int):
            num_encoder_layers = [num_encoder_layers] * self.cross_num
        if isinstance(t_length,int):
            t_length = [t_length] * self.cross_num
        self.base_encoders = nn.ModuleList([TransformerEncoder(seq_length=t_length[i], embed_dim=embedding_dims[i] or dims[i], depth=num_encoder_layers[i],
                                                    num_heads=encoder_heads[i], head_dim=encoder_head_dims[i], mlp_ratio=dim_ff_ratio,
                                                    drop_ratio=dropout, attn_drop_ratio=dropout,
                                                    drop_path_ratio=dropout, act_layer=activation)
                                            for i in range(self.cross_num)])
        if attention is not None and encoder is None:
            self.fusion = AttentionEncoder(cross_num=self.cross_num, embed_dim=d_model, depth=num_decoder_layers, num_heads=nhead,
                                              mlp_ratio=dim_ff_ratio, drop_ratio=dropout, attn_drop_ratio=dropout,
                                              drop_path_ratio=dropout, act_layer=activation, attention=attention)
        elif encoder:
            self.fusion = encoder(cross_num=self.cross_num, embed_dim=d_model, depth=num_decoder_layers, num_heads=nhead,
                                  mlp_ratio=dim_ff_ratio, drop_ratio=dropout, attn_drop_ratio=dropout,
                                  drop_path_ratio=dropout, act_layer=activation)
        else:
            self.fusion = None
        self.fusion_method = fusion
        self.apply(_init_weights)

    def forward(self, *srcs): #text video audio
        # embeding
        srcs_, masks = [], []
        for i in range(self.cross_num):
            src = self.src_linears[i](srcs[i])
            srcs_.append(src)
            mask = None
            if self.mask is not None:
                mask = (srcs[i].sum(-1)!=0).float()
            masks.append(mask)

        # feature extract
        feats_ = []
        for i in range(self.cross_num):
            feats = self.base_encoders[i](srcs_[i], masks[i])
            if not self.aligned:
                feats = feats[:,0]
            feats_.append(feats)

        # features projection aligned
        # unimodel features logits
        logits_, feats__, heads_ = [], [], []
        for i in range(self.cross_num):
            feat_ = feats_[i]
            if self.aligned:
                feat_ = feat_[:,0]
            # head = self.heads[i](feat_)
            head_ = self.head(feat_)
            heads_.append(head_)
            logit = torch.sigmoid(head_)
            logits_.append(logit)
            feats__.append(feat_)

        # multimodel fusion
        if self.fusion:
            ss = self.fusion(*feats_)
            if self.aligned:
                ss = [s[:,0] for s in ss]
        else:
            ss = feats_

        # s = self.fusion_method(ss)
        # head = self.head(s)

        # shared classification head
        heads = [self.head(s) for s in ss]
        # head = sum(heads)

        # head = [1 / (1 + torch.exp(-head)).detach() * head for head in heads]
        # head = sum(head)

        # classification head latefusion
        head = self.fusion_method(heads)

        logits = torch.sigmoid(head)

        return_dict = {
            "logits_": logits_,
            "feats_": feats__,
            "heads_": heads_,
            "feats": ss,
            "head": head,
            "heads": heads,
        }

        return logits, return_dict


def WLoss(return_dict, labels, alpha_pos=3, beta_pos=2.5, alpha_neg=2, beta_neg=2):
    heads = return_dict["heads"]
    binary_label = (labels > 0).int()
    pos_neg_label = 2 * (binary_label - 0.5)

    power_alpha = (binary_label * (alpha_pos - alpha_neg) + alpha_neg)
    power_beta = (binary_label * (beta_pos - beta_neg) + beta_neg)

    logits = [((1 - (1 / (1 + torch.exp(- head * pos_neg_label))) ** power_alpha) ** power_beta).detach() * head for head in heads]

    head = sum(logits)
    return head



cos_d = lambda x, y: F.cosine_similarity(x, y).mean()


def d_reg(feats, feat, d_fun=cos_d):
    loss = 0
    for feat_ in feats:
        loss += d_fun(feat_, feat)
    return loss


def wfeats(feats, feat, model):
    d = [F.cosine_similarity(feat_, feat).unsqueeze(-1) for feat_ in feats]
    d = - torch.stack(d)
    d = torch.exp(d)
    sim = d / d.sum(0, keepdim=True) * len(feats)
    feats_ = [feat_ * sim[i].detach() for i, feat_ in enumerate(feats)]

    if model.fusion:
        ss = model.fusion(*feats_)
        if model.aligned:
            ss = [s[:,0] for s in ss]
    else:
        ss = feats_

    heads = [model.head(s) for s in ss]

    # classification head latefusion
    head = model.fusion_method(heads)

    logits = torch.sigmoid(head)

    return_dict = {
        "feats": ss,
        "head": head,
        "heads": heads,
    }

    return logits, return_dict





def lq_loss(x, labels, q_pos_h=0.05, q_pos_e=0.1, q_neg_h=1, q_neg_e=1, threshold=0.5, gamma_pos=1, gamma_neg=1):
    # q_pos_h, q_pos_e, q_neg_h, q_neg_e = 0.05, 0.1, 0.1, 1
    binary_label = (labels > 0).int()
    p = torch.sigmoid(x)
    mask = (p > threshold).int()
    q = (q_pos_h + mask * (q_pos_e - q_pos_h)) * binary_label + (1 - binary_label) * (q_neg_h + (1 - mask) * (q_neg_e - q_neg_h))
    loss = labels * binary_label * (1 - p ** q) / q * gamma_pos + (1 - labels) * (1 - binary_label) * (1 - (1 - p) ** q) / q * gamma_neg
    loss = loss.sum(-1)
    return loss.mean()


def MLPModel(*args, **kargs):
    return BaseAttentionModel(*args, **kargs, attention=MLPFusion)



if __name__ == '__main__':
    torch.sigmoid()

