"""DRAGON convolutional network for astronomical image cutouts."""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

DRAGON_CUTOUT_SIZE = 96

# Three symmetric 2x downsampling stages.
DRAGON_SIZE_DIVISOR = 8

# Fixed 0.5, 1, 2, and 4 arcsec bins at 0.168 arcsec/pixel. Whole-pixel values
# keep the innermost edge at the stride-2 sampling limit.
DRAGON_RING_EDGES_PX = (3.0, 6.0, 12.0, 24.0)

# Drop ring edges too small to be sampled on a feature map.
DRAGON_MIN_RING_EDGE_MAP_PX = 1.5

# At 48 pixels, every stage retains a non-empty outer ring.
DRAGON_MIN_SIZE = 48

# Shared checkpoint keys used by the channel-conversion script.
DRAGON_FIRST_CONV_WEIGHT_KEY = "layer1.0.0.weight"
DRAGON_CLASSIFIER_BIAS_KEY = "classifier.bias"
DRAGON_HEAD_PREFIXES = ("feature_norm.", "mixer.", "classifier.")


def _stage_ring_edges(stride):
    """Ring edges in feature-map pixels for a stage at ``stride``."""
    edges = tuple(
        edge / stride
        for edge in DRAGON_RING_EDGES_PX
        if edge / stride >= DRAGON_MIN_RING_EDGE_MAP_PX
    )
    if not edges:
        raise ValueError(f"No ring edge survives stride {stride}.")
    return edges


class BlurPool2d(nn.Module):
    """Downsample by 2 with a replicate-padded ``[1, 3, 3, 1]`` blur."""

    def __init__(self, channels):
        super().__init__()
        weights = torch.tensor([1.0, 3.0, 3.0, 1.0])
        kernel = torch.outer(weights, weights)
        kernel = kernel / kernel.sum()
        # The fixed filter is not part of checkpoints.
        self.register_buffer(
            "blur", kernel[None, None].repeat(channels, 1, 1, 1), persistent=False
        )
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.blur, stride=2, groups=self.channels)


class Downsample(nn.Module):
    """Apply a non-in-place activation followed by ``BlurPool2d``."""

    def __init__(self, channels):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.pool = BlurPool2d(channels)

    def forward(self, x):
        return self.pool(self.act(x))


def _conv_bn(in_channels, out_channels, kernel_size=3):
    """Build a bias-free convolution and BatchNorm pair.

    Replicate padding avoids both a dark artificial border and reflected source
    copies.
    """
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]


def _conv_bn_act(in_channels, out_channels, kernel_size=3):
    """Build a convolution, BatchNorm, and LeakyReLU block."""
    return nn.Sequential(
        *_conv_bn(in_channels, out_channels, kernel_size),
        nn.LeakyReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an unactivated identity sum.

    Zero-initializing the second BatchNorm scale makes the block start as the
    identity. Consumers apply their own activation.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1, padding_mode="replicate", bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1, padding_mode="replicate", bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + x


class RingPool(nn.Module):
    """Pool each channel over concentric rings.

    Every ring contributes a signed mean. Maxima come from ``max_input`` and
    omit the outer ring unless ``outer_max`` is enabled. Geometry is cached by
    map shape, device, and dtype.
    """

    def __init__(self, edges, *, outer_max=False):
        super().__init__()
        edges = tuple(float(edge) for edge in edges)
        if not edges:
            raise ValueError("RingPool needs at least one edge.")
        if any(edge <= 0 for edge in edges):
            raise ValueError(f"RingPool edges must be positive; received {edges}.")
        if any(b <= a for a, b in pairwise(edges)):
            raise ValueError(f"RingPool edges must increase; received {edges}.")
        self.edges = edges
        self.n_rings = len(edges) + 1
        # Fixing max rings by index keeps the output width size-independent.
        self.n_max_rings = self.n_rings if outer_max else self.n_rings - 1
        self._cache = {}

    def out_features(self, channels):
        return channels * (self.n_rings + self.n_max_rings)

    def geometry(self, height, width, device=None, dtype=torch.float32):
        """Return ``(mean_weights, ring_indices)`` for one map size, cached."""
        key = (height, width, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
        cols = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2
        radius = (rows[:, None] ** 2 + cols[None, :] ** 2).sqrt().flatten()

        bounds = (0.0,) + self.edges + (float("inf"),)
        weights, indices = [], []
        for index in range(self.n_rings):
            low, high = bounds[index], bounds[index + 1]
            inside = (radius >= low) & (radius < high)
            count = int(inside.sum())
            if count == 0:
                raise ValueError(
                    f"RingPool ring {index} (radii [{low}, {high}) feature-map "
                    f"pixels) is empty on a {height}x{width} map."
                )
            weights.append(inside.to(dtype) / count)
            indices.append(inside.nonzero(as_tuple=True)[0])
        geometry = (torch.stack(weights), tuple(indices))
        self._cache[key] = geometry
        return geometry

    def forward(self, x, max_input=None):
        """``x`` is pooled by mean; ``max_input`` (default ``x``) by max."""
        if max_input is None:
            max_input = x
        height, width = x.shape[-2:]
        weights, indices = self.geometry(height, width, x.device, x.dtype)
        means = x.flatten(2) @ weights.t()
        flat_max = max_input.flatten(2)
        maxes = torch.stack(
            [flat_max[..., indices[ring]].amax(-1) for ring in range(self.n_max_rings)],
            dim=-1,
        )
        return torch.cat([means, maxes], dim=-1).flatten(1)


class Stage(nn.Module):
    """Downsample, project to a new width, and apply one residual block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = Downsample(in_channels)
        self.project = _conv_bn_act(in_channels, out_channels)
        self.block = ResidualBlock(out_channels)

    def forward(self, x):
        return self.block(self.project(self.down(x)))


class RingTap(nn.Module):
    """Ring-pool signed means and rectified maxima from one stage."""

    def __init__(self, edges, *, outer_max=False):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.pool = RingPool(edges, outer_max=outer_max)

    def out_features(self, channels):
        return self.pool.out_features(channels)

    def forward(self, x):
        return self.pool(x, self.act(x))


class DRAGON(nn.Module):
    """Four-stage, multi-scale CNN for astronomical cutout classification.

    ``channels`` and ``num_classes`` must match the dataset and label mapping.
    ``widths`` controls the backbone; ``head_width`` controls the classifier.
    """

    def __init__(
        self,
        *,
        channels,
        num_classes,
        widths=(24, 64, 128, 96),
        head_width=96,
        dropout=0.4,
    ):
        super().__init__()
        if len(widths) != 4:
            raise ValueError(f"widths must name four stages; received {widths}.")
        stem, first, second, third = widths

        # Preserve an 11-pixel receptive field before the first downsampling.
        # The final BatchNorm stays unactivated for downstream consumers.
        self.layer1 = nn.Sequential(
            _conv_bn_act(channels, stem, kernel_size=5),
            _conv_bn_act(stem, stem),
            _conv_bn_act(stem, stem),
            nn.Sequential(*_conv_bn(stem, stem)),
        )
        self.layer2 = Stage(stem, first)
        self.layer3 = Stage(first, second)
        self.layer4 = Stage(second, third)

        # Tap every scale so tight pairs and broad tidal features both reach
        # the classifier head.
        stage_strides = (1, 2, 4, 8)
        edges = [_stage_ring_edges(stride) for stride in stage_strides]
        self.tap1 = RingTap(edges[0])
        self.tap2 = RingTap(edges[1])
        # Coarse outer-ring maxima retain localized tidal features.
        self.tap3 = RingTap(edges[2], outer_max=True)
        self.tap4 = RingTap(edges[3], outer_max=True)

        taps = (self.tap1, self.tap2, self.tap3, self.tap4)
        feature_dim = sum(
            tap.out_features(width)
            for tap, width in zip(taps, (stem, first, second, third))
        )
        # Normalize mixed mean/max statistics before the shared head.
        self.feature_norm = nn.BatchNorm1d(feature_dim)
        self.drop = nn.Dropout(dropout)
        self.mixer = nn.Sequential(
            nn.Linear(feature_dim, head_width),
            nn.BatchNorm1d(head_width),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier = nn.Linear(head_width, num_classes)

        self._initialize_weights()
        self._verify_checkpoint_keys()
        self._verify_ring_geometry(stage_strides)

    def _verify_checkpoint_keys(self):
        """Fail at construction if an anchor stops naming part of the model."""
        anchors = (
            DRAGON_FIRST_CONV_WEIGHT_KEY,
            DRAGON_CLASSIFIER_BIAS_KEY,
            *DRAGON_HEAD_PREFIXES,
        )
        names = set(self.state_dict())
        missing = sorted(
            anchor
            for anchor in anchors
            if not any(name.startswith(anchor) for name in names)
        )
        if missing:
            raise RuntimeError(
                "DRAGON checkpoint anchor(s) no longer name any state entry; "
                "update the DRAGON_* constants alongside the architecture: "
                f"{missing}"
            )

    def _verify_ring_geometry(self, stage_strides):
        """Fail at construction, not at the first forward, if a ring is empty.

        The smallest legal input is the worst case for the outer rings, so
        checking it once here covers every input this model will accept.
        """
        taps = (self.tap1, self.tap2, self.tap3, self.tap4)
        for tap, stride in zip(taps, stage_strides):
            size = DRAGON_MIN_SIZE // stride
            try:
                tap.pool.geometry(size, size)
            except ValueError as error:
                raise ValueError(
                    f"Ring geometry is unusable at the minimum input size "
                    f"{DRAGON_MIN_SIZE} (stride {stride} gives a {size}x{size} "
                    f"map): {error}"
                ) from error
            # Forward passes use device-specific cache entries.
            tap.pool._cache.clear()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

        # The hidden layer uses Kaiming initialization; the classifier stays
        # near zero under the small-normal rule above.
        nn.init.kaiming_normal_(self.mixer[0].weight, a=0.01, nonlinearity="leaky_relu")
        nn.init.zeros_(self.mixer[0].bias)

        # Start residual blocks as identities.
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
        expected_channels = self.layer1[0][0].in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"DRAGON was built for {expected_channels} channel(s); "
                f"received {x.shape[1]}."
            )

        stem = self.layer1(x)
        mid = self.layer2(stem)
        out = self.layer3(mid)
        deep = self.layer4(out)

        features = torch.cat(
            [self.tap1(stem), self.tap2(mid), self.tap3(out), self.tap4(deep)], dim=1
        )
        return self.classifier(self.mixer(self.drop(self.feature_norm(features))))
