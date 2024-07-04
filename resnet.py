import torch as t
import torch.nn as nn

class ResidualBlock(t.nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block as per https://arxiv.org/pdf/1512.03385
        """
        super().__init__()
        self.left = nn.Sequential(*[
            nn.Conv2d(in_feats, out_feats, 3, first_stride, 1),
            nn.BatchNorm2d(out_feats),
            nn.ReLU(),
            nn.Conv2d(out_feats, out_feats, 3, 1, 1),
            nn.BatchNorm2d(out_feats)
            ])

        if first_stride == 1:
            self.right = nn.Identity()
        else:
            self.right = nn.Sequential(*[
                nn.Conv2d(in_feats, out_feats, 1, first_stride),
                nn.BatchNorm2d(out_feats)
                ])

        self.relu = nn.ReLU()

        def forward(self, x: t.Tensor) -> t.Tensor:
            """
            x: shape (batch, in_feats, height, width)

            Return: shape (batch, out_feats, height / stride, width / stride)

            If no downsampling block is present, the residual branch (right branch) 
            adds the input to the output. If there is downsampling, then the residual
            (skip) connection includes a convolution and a batch norm.
            """
            return self.relu(self.left(x) + self.right(x))


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()

        blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return t.mean(x, (2, 3))


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.comps = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            *[BlockGroup(n, in_features, out_features, first_stride)
                for n, in_features, out_features, first_stride in zip(
                  n_blocks_per_group,
                  [64] + out_features_per_group,
                  out_features_per_group,
                  first_strides_per_group)
            ],
            AveragePool(),
            nn.Linear(out_features_per_group[-1], n_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        return self.comps(x)
