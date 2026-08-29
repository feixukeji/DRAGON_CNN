import torch
from torch import nn
from torch.nn import functional as F

DRAGON_CUTOUT_SIZE = 96

# Four 2x stages, so every stage sees an even-sized map and samples it symmetrically.
DRAGON_SIZE_DIVISOR = 16

# Divisibility alone is not enough. The reflection pad in MaxBlurPool2d needs a
# map of at least 3 to pad by one, and the fourth stage sees one eighth of the
# input, so a 16x16 cutout would clear the divisor and then raise inside F.pad.
DRAGON_MIN_SIZE = 32

# Single-sourced checkpoint keys. Scripts that rewrite a saved state_dict import
# these instead of hard-coding module paths, and __init__ verifies that they
# still name real parameters, so renaming a layer fails loudly at construction
# rather than silently breaking checkpoint conversion.
DRAGON_FIRST_CONV_WEIGHT_KEY = "layer1.0.weight"
DRAGON_CLASSIFIER_WEIGHT_KEY = "classifier.weight"
DRAGON_CLASSIFIER_BIAS_KEY = "classifier.bias"
DRAGON_CHECKPOINT_KEYS = (
    DRAGON_FIRST_CONV_WEIGHT_KEY,
    DRAGON_CLASSIFIER_WEIGHT_KEY,
    DRAGON_CLASSIFIER_BIAS_KEY,
)


class MaxBlurPool2d(nn.Module):
    """Anti-aliased 2x downsample that keeps point-source peaks.

    Average pooling is a low-pass filter, so it erases the saddle between two
    close PSFs -- the one feature that separates a pair from a single source.
    Plain strided max pooling keeps the peak but aliases it. This takes the
    stride-1 max first, blurs with a binomial kernel, then subsamples, which
    preserves the peak and stays shift-robust (Zhang 2019).

    The (k=2, stride=1) max followed by a 3-tap stride-2 blur halves an
    even-sized map exactly and samples it symmetrically, so the eight dihedral
    training augmentations stay equivalent views of the same object.

    The blur pads by reflection rather than with zeros. Zero padding would give
    the border a fraction of the kernel mass -- 3/4 along an edge, 9/16 in a
    corner -- which compounds over four stages into a fixed vignette on the
    final map. BatchNorm rescales a whole channel at once and cannot undo a
    gradient that varies across the map, so the network would have to spend
    capacity compensating. Reflection keeps unit gain everywhere and, being a
    symmetric boundary condition, preserves the dihedral equivalence exactly.
    """

    def __init__(self, channels):
        super().__init__()
        weights = torch.tensor([1.0, 2.0, 1.0])
        kernel = torch.outer(weights, weights)
        kernel = kernel / kernel.sum()
        # A fixed filter, so keep it out of the checkpoint.
        self.register_buffer(
            "blur", kernel[None, None].repeat(channels, 1, 1, 1), persistent=False
        )
        self.channels = channels

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.blur, stride=2, groups=self.channels)


def _conv_bn_act(in_channels, out_channels):
    """Convolution without bias, since the BatchNorm shift subsumes it."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    ]


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions on a clean identity shortcut.

    Nothing follows the sum. Paired with the zeroed ``bn2.weight`` set in
    ``DRAGON._initialize_weights`` the block is then exactly the identity at
    init, and the shortcut carries gradients unscaled.

    A LeakyReLU after the sum would break both properties. The incoming tensor
    is already an activation, so a second LeakyReLU multiplies its negative
    tail by another 0.01 -- three more such factors along a path that stays
    negative, on top of the 0.01 each intervening stage already applies, which
    is to say LeakyReLU degenerates into ReLU exactly where its leak was
    supposed to matter -- and it scales shortcut gradients by 0.01 wherever the
    sum is negative. Post-sum activation costs a ReLU ResNet nothing at init,
    because there the shortcut carries non-negative values and ``relu(x) == x``
    leaves the identity intact; that equivalence does not survive the switch to
    LeakyReLU. The next stage supplies the nonlinearity.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(out)) + x


class DRAGON(nn.Module):
    """Eight-block CNN for close-pair classification in survey cutouts.

    The resolution schedule is set by the separations the network has to
    resolve: pairs span roughly 2-25 pixels, so the first block runs at full
    resolution and the map is still 24x24 at ``layer4``, where the pair is
    several feature-map pixels wide. Widths double as the map halves, keeping
    the per-stage cost roughly flat.

    ``layer1`` .. ``layer8`` are the unfreezing units used by transfer
    learning, and ``layer4`` is the Grad-CAM target.

    ``channels`` and ``num_classes`` are required: they must agree with the
    stored cutouts and with ``labels.csv``, and a default that silently
    disagreed with either would surface only as a shape error much later.
    """

    def __init__(self, *, channels, num_classes, dropout=0.5):
        super().__init__()

        self.layer1 = nn.Sequential(*_conv_bn_act(channels, 48))
        self.layer2 = nn.Sequential(MaxBlurPool2d(48), *_conv_bn_act(48, 64))
        self.layer3 = ResidualBlock(64)
        self.layer4 = nn.Sequential(MaxBlurPool2d(64), *_conv_bn_act(64, 128))
        self.layer5 = ResidualBlock(128)
        self.layer6 = nn.Sequential(MaxBlurPool2d(128), *_conv_bn_act(128, 256))
        self.layer7 = ResidualBlock(256)
        self.layer8 = nn.Sequential(MaxBlurPool2d(256), *_conv_bn_act(256, 512))

        # Global pooling instead of a flatten, so the classifier does not
        # hard-code the input size and does not carry a 2M-parameter hidden layer.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()
        self._verify_checkpoint_keys()

    def _verify_checkpoint_keys(self):
        parameter_names = {name for name, _ in self.named_parameters()}
        missing = sorted(set(DRAGON_CHECKPOINT_KEYS) - parameter_names)
        if missing:
            raise RuntimeError(
                "DRAGON checkpoint key(s) no longer name a parameter; update the "
                f"DRAGON_*_KEY constants alongside the architecture: {missing}"
            )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

        # Start every residual branch at zero so each block begins as the
        # identity. Exact here, because no activation follows the sum.
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x):
        height, width = x.shape[-2:]
        if (
            height % DRAGON_SIZE_DIVISOR
            or width % DRAGON_SIZE_DIVISOR
            or height < DRAGON_MIN_SIZE
            or width < DRAGON_MIN_SIZE
        ):
            raise ValueError(
                "DRAGON input height and width must be multiples of "
                f"{DRAGON_SIZE_DIVISOR} and at least {DRAGON_MIN_SIZE}; "
                f"received {height}x{width}."
            )

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = torch.flatten(self.gap(out), 1)
        out = self.drop(out)
        return self.classifier(out)
