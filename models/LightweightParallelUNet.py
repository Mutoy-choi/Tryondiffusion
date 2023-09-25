import torch
from torch import nn


class LightweightParallelUNet(nn.Module):
    def __init__(self, emb_dim, config) -> None:
        super().__init__()
        self.garment_unet = LightweightGarmentUNet(emb_dim, config['garment_unet'])
        self.person_unet = LightweightPersonUNet(emb_dim, config['person_unet'])

    def forward(self, x, gar_emb, pose_emb, seg_garment):
        g_feature = self.garment_unet(seg_garment, gar_emb)
        deno_x = self.person_unet(x, g_feature, pose_emb)

        return deno_x



class GarmentUNet(nn.Module):
    def __init__(self, emb_dim, config, img_channel=3) -> None:
        super().__init__()

        unet_in_channel = config['dstack']['blocks'][0]['channels']

        self.conv1 = nn.Conv2d(img_channel, unet_in_channel, 3, padding=1)  # channels
        self.dstack = DStack(emb_dim, config['dstack'])
        self.ustack = UStack(emb_dim, config['ustack'])

    def forward(self, x, emb):

        x = self.conv1(x)
        x, d_list = self.dstack(x, emb, None)
        x, u_list = self.ustack(x, d_list, emb, None)

        return d_list, u_list


class PersonUNet(nn.Module):
    def __init__(self, emb_dim, config, img_channel=3) -> None:
        super().__init__()

        unet_channel = config['dstack']['blocks'][0]['channels']

        self.conv1 = nn.Conv2d(img_channel, unet_channel, 3, padding=1)  # channels
        self.dstack = DStack(emb_dim, config['dstack'])
        self.ustack = UStack(emb_dim, config['ustack'])
        self.conv2 = nn.Conv2d(unet_channel, img_channel, 3, padding=1)


    def forward(self, x, g_feature, emb):
        garment_d, garment_u = g_feature
        x = self.conv1(x)

        x, d_list = self.dstack(x, emb, garment_d)
        x, u_list = self.ustack(x, d_list, emb, garment_u)

        x = self.conv2(x)

        return x


class LightweightGarmentUNet(GarmentUNet):
    # Configuration for the lightweight model
    def __init__(self, emb_dim, config, img_channel=3) -> None:
        # Reduce the channel sizes by a factor of 2 (or any other factor you consider suitable)
        for block in config['dstack']['blocks']:
            block['channels'] = block['channels'] // 2
        for block in config['ustack']['blocks']:
            block['channels'] = block['channels'] // 2
        super().__init__(emb_dim, config, img_channel=img_channel)

    def forward(self, x, emb):

        x = self.conv1(x)
        x, d_list = self.dstack(x, emb, None)
        x, u_list = self.ustack(x, d_list, emb, None)

        return d_list, u_list

class LightweightPersonUNet(PersonUNet):
    # Configuration for the lightweight model
    def __init__(self, emb_dim, config, img_channel=3) -> None:
        # Reduce the channel sizes by a factor of 2 (or any other factor you consider suitable)
        for block in config['dstack']['blocks']:
            block['channels'] = block['channels'] // 2
        for block in config['ustack']['blocks']:
            block['channels'] = block['channels'] // 2
        super().__init__(emb_dim, config, img_channel=img_channel)

    def forward(self, x, g_feature, emb):
        garment_d, garment_u = g_feature
        x = self.conv1(x)

        x, d_list = self.dstack(x, emb, garment_d)
        x, u_list = self.ustack(x, d_list, emb, garment_u)

        x = self.conv2(x)

        return x

class DStack(nn.Module):
    def __init__(self, emb_dim, config: dict) -> None:
        super().__init__()

        self.rep_blocks = nn.ModuleList([])
        self.block_types = []
        for block_config in config['blocks']:
            channels = block_config.get('channels')
            block_type = block_config.get('block_type', 'FiLM_ResBlk')

            module = nn.ModuleList([
                RepeatBlocks(emb_dim, block_config),
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)])

            self.rep_blocks.append(module)
            self.block_types.append(block_type)

    def forward(self, x, emb, cross_list=None):
        d_list = []

        for i, ((rep_block, down), block_type) in enumerate(zip(self.rep_blocks, self.block_types)):
            if block_type == 'FiLM_ResBlk_Self_Cross':
                x = rep_block(x, emb, cross_list[i])
            else:
                x = rep_block(x, emb)

            d_list.append(x)
            x = down(x)

        return x, d_list


class UStack(nn.Module):
    def __init__(self, emb_dim, config) -> None:
        super().__init__()

        self.rep_blocks = nn.ModuleList([])
        self.block_types = []
        for block_config in config['blocks']:
            channels = block_config.get('channels')
            block_type = block_config.get('block_type', 'FiLM_ResBlk')

            module = nn.ModuleList([
                nn.ConvTranspose2d(channels * 2, channels, 2, stride=2),
                nn.Conv2d(channels * 2, channels, 1),
                RepeatBlocks(emb_dim, block_config)
            ])

            self.rep_blocks.append(module)

            self.block_types.append(block_type)

    def forward(self, x, d_list, emb, cross_list=None):

        u_list = []
        d_list = reversed(d_list)
        iter_blocks = zip(self.rep_blocks, d_list, self.block_types)
        for i, ((upconv, combine, rep_block), skip, block_type) in enumerate(iter_blocks):

            x = upconv(x)

            con = torch.concat([x, skip], dim=1)
            x = combine(con)

            if block_type == 'FiLM_ResBlk_Self_Cross':
                x = rep_block(x, emb, cross_list[i])
            else:
                x = rep_block(x, emb)

            u_list.append(x)

        return x, u_list


class RepeatBlocks(nn.Module):
    def __init__(self, emb_dim, config: dict) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([])

        block_type = config.pop('block_type', 'FiLM_ResBlk')
        repeat = config.pop('repeat', 1)

        Block_CLS = None
        if block_type == "FiLM_ResBlk":
            Block_CLS = FiLMResBlk
        elif block_type == "FiLM_ResBlk_Self_Cross":
            Block_CLS = FiLMResBlkSelfCross
        else:
            raise Exception('It is not support blocks type.')

        for _ in range(repeat):
            self.blocks.append(Block_CLS(emb_dim, **config))

    def forward(self, x, emb, cross=None):
        for blk in self.blocks:
            if blk.__class__ is FiLMResBlkSelfCross:
                x = blk(x, emb, cross)
            else:
                x = blk(x, emb)
        return x


class FiLMResBlk(nn.Module):
    def __init__(self, emb_dim, channels, kernel_size=3) -> None:
        super().__init__()

        self.film_layer = FiLM(emb_dim, channels)
        self.resblock = ResBlock(channels, kernel_size)

    def forward(self, x, emb):
        x = self.film_layer(x, emb)
        x = self.resblock(x)

        return x


class FiLMResBlkSelfCross(nn.Module):
    def __init__(self, emb_dim, channels, kernel_size=3, num_heads=8) -> None:
        super().__init__()

        self.film_layer = FiLM(emb_dim, channels)
        self.resblock = ResBlock(channels, kernel_size)
        self.selfattn = SelfAttention(channels, num_heads)
        self.crossattn = CrossAttention(channels, num_heads)

    def forward(self, x, emb, cross):
        x = self.film_layer(x, emb)
        x = self.resblock(x)

        x = self.selfattn(x)
        x = self.crossattn(x, cross)

        return x


class FiLM(nn.Module):
    def __init__(self, emb_dim, channels) -> None:
        super().__init__()

        self.fc_scale = nn.Linear(emb_dim, channels)
        self.fc_bias = nn.Linear(emb_dim, channels)

    def forward(self, x, emb):
        batch_size, _, _, _ = x.size()
        emb_flat = emb.view(batch_size, -1)  # Flatten the spatial dimensions
        scale = self.fc_scale(emb_flat).unsqueeze(-1).unsqueeze(-1)
        bias = self.fc_bias(emb_flat).unsqueeze(-1).unsqueeze(-1)

        return x * scale + bias


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        group = min(channels // 4, 32)
        self.nomr1 = nn.GroupNorm(group, channels)
        self.swish1 = nn.SiLU()
        self.conv2d1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)

        self.nomr2 = nn.GroupNorm(group, channels)
        self.swish2 = nn.SiLU()
        self.conv2d2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        skip = x
        x = self.nomr1(x)
        x = self.swish1(x)
        x = self.conv2d1(x)

        x = self.nomr2(x)
        x = self.swish2(x)
        x = self.conv2d2(x)

        return x + skip


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4) -> None:
        super().__init__()

        group = min(channels // 4, 32)
        self.norm1 = nn.GroupNorm(group, channels)

        self.head_channels = channels
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(channels, num_heads)
        self.norm2 = nn.GroupNorm(group, channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, mlp_ratio * channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mlp_ratio * channels, channels, kernel_size=1)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        assert self.head_channels == channels

        x = self.norm1(x)

        emb_x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        attn_out, _ = self.attn(emb_x, emb_x, emb_x)

        attn_out = attn_out.permute(1, 2, 0).contiguous().view(batch_size, channels, height, width)

        x = self.norm2(attn_out + x)
        out = self.ffn(x)

        return x + out


class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4) -> None:
        super().__init__()

        group = min(channels // 4, 32)
        self.norm1 = nn.GroupNorm(group, channels)

        self.channels = channels
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(channels, num_heads)
        self.norm2 = nn.GroupNorm(group, channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, mlp_ratio * channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mlp_ratio * channels, channels, kernel_size=1)
        )

    def forward(self, x_q, x_kv):
        batch_size, _, height_query, width_query = x_q.size()
        batch_size_kv, _, height_kv, width_kv = x_kv.size()

        assert batch_size == batch_size_kv

        x_q = self.norm1(x_q)

        emb_x_q = x_q.view(batch_size, self.channels, -1).permute(0, 2, 1)
        emb_x_kv = x_kv.view(batch_size, self.channels, -1).permute(0, 2, 1)

        attn_out, _ = self.attn(emb_x_q, emb_x_kv, emb_x_kv)

        attn_out = attn_out.permute(1, 2, 0).contiguous().view(batch_size, self.channels, height_query, width_query)

        x_q = self.norm2(attn_out + x_q)
        out = self.ffn(x_q)

        return x_q + out


def debug():
    IMG_CHANNEL = 3

    EMB_DIM = 128

    parallel_config = {
        'garment_unet': {
            'dstack': {
                'blocks': [
                    {
                        'channels': 128,
                        'repeat': 3
                    },
                    {
                        'channels': 256,
                        'repeat': 4
                    },
                    {
                        'channels': 512,
                        'repeat': 6
                    },
                    {
                        'channels': 1024,
                        'repeat': 7
                    }]
            },
            'ustack': {
                'blocks': [
                    {
                        'channels': 1024,
                        'repeat': 7
                    },
                    {
                        'channels': 512,
                        'repeat': 6
                    }]
            }
        },
        'person_unet': {
            'dstack': {
                'blocks': [
                    {
                        'channels': 128,
                        'repeat': 3
                    },
                    {
                        'channels': 256,
                        'repeat': 4
                    },
                    {
                        'block_type': 'FiLM_ResBlk_Self_Cross',
                        'channels': 512,
                        'repeat': 6
                    },
                    {

                        'block_type': 'FiLM_ResBlk_Self_Cross',
                        'channels': 1024,
                        'repeat': 7
                    }]
            },
            'ustack': {
                'blocks': [
                    {
                        'block_type': 'FiLM_ResBlk_Self_Cross',
                        'channels': 1024,
                        'repeat': 7
                    },
                    {
                        'block_type': 'FiLM_ResBlk_Self_Cross',
                        'channels': 512,
                        'repeat': 6
                    },
                    {
                        'channels': 256,
                        'repeat': 4
                    },
                    {
                        'channels': 128,
                        'repeat': 3
                    }]
            }
        }
    }

    x = torch.randn(1, IMG_CHANNEL, 128, 128)
    garment_x = torch.randn(1, IMG_CHANNEL, 128, 128)
    emb = torch.randn(128)

    parallel_unet = ParallelUNet(EMB_DIM, parallel_config)
    print(parallel_unet(x, emb, garment_x))

    pytorch_total_params = sum(p.numel() for p in parallel_unet.parameters() if p.requires_grad)

    print(pytorch_total_params)

if __name__ == '__main__':
    debug()

