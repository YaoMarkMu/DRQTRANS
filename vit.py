import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models import vit_tiny_patch6_84
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.image_height, self.image_width = image_height, image_width
        # print(image_height, image_width)
        # print(patch_height, patch_width)
        # print(image_height % patch_height)
        # print(image_width % patch_width == 0)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.policy_token = nn.Parameter(torch.randn(1, 1, dim))
        self.reconstruct_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


    def forward(self, img):
        img = img.reshape(-1, 3, self.image_height, self.image_width)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        policy_tokens = repeat(self.policy_token, '() n d -> b n d', b = b)
        reconstruct_tokens = repeat(self.reconstruct_token, '() n d -> b n d', b=b)
        x = torch.cat((policy_tokens, x, reconstruct_tokens), dim=1)
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)
        x = self.transformer(x)
        policy_latent = x[:, 0]
        reconstruct_latent = x[:, -1]
        return policy_latent, reconstruct_latent

class DiT(nn.Module):
    def __init__(self, num_step, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_step + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.reconstruct_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, img_latent):
        b, n, _ = img_latent.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, img_latent), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        return x[:, 0]

class RiT(nn.Module):
    def __init__(self, num_step, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_step , dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, img_latent):
        b, n, _ = img_latent.shape
        img_latent += self.pos_embedding[:, :n]
        x = self.transformer(img_latent)
        return x
class T_Encoder(nn.Module):
    def __init__(self,obs_shape, feature_dim):
        super().__init__()
        self.num_step=int(obs_shape[0]/3)
        self.hidden_dim=512
        #self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.image_encode = ViT(
        image_size=obs_shape[-1],
        patch_size=12,
        dim=self.hidden_dim,
        depth=6,
        heads=16,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
        self.policy_encoder = DiT(
        num_step=self.num_step,
        dim=256,
        depth=6,
        heads=16,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
        self.reconstruct_encoder = RiT(
        num_step=self.num_step,
        dim=self.hidden_dim,
        depth=6,
        heads=16,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_dim),
        )


    def forward(self,img_sequence):
        latent = self.image_encode(img_sequence)
        rec_latent = latent[0].reshape(-1, self.num_step, self.hidden_dim)
        policy_latent = latent[1].reshape(-1, self.num_step, self.hidden_dim)
        policy_feature = self.mlp(self.policy_encoder(policy_latent))
        rec_feature = self.reconstruct_encoder(rec_latent)
        return policy_feature

class Timm_Encoder(nn.Module):
    def __init__(self,obs_shape, feature_dim):
        super().__init__()
        self.num_step=int(obs_shape[0]/3)
        self.feature_dim = feature_dim
        self.image_encode = vit_tiny_patch6_84()
        self.policy_encoder = DiT(
        num_step=self.num_step,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1
    )
        self.mlp = nn.Sequential(
            nn.Linear(192, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_dim),
        )
        self.con_mlp = nn.Sequential(
            nn.Linear(192, 256),
            nn.GELU(),
            nn.Linear(256, self.feature_dim),
        )
    def forward(self,img_sequence,detach):
        # img_sequence = img_sequence.reshape(-1,3,84,84)
        latent = self.image_encode.forward_features(img_sequence)
        # print(latent[0].shape)
        # policy_latent = latent[1].reshape(-1, self.num_step*192)
        policy_feature = self.con_mlp(latent)
        #print(policy_feature.shape)
        # policy_feature = self.mlp(self.policy_encoder(policy_latent))
        if detach:
            policy_feature=policy_feature.detach()

        return policy_feature
