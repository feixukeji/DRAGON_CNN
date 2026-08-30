"""DRAGON: close-pair classification in HSC cutouts.

The resolution schedule is derived from the scales the labels are made of
rather than from a generic classifier backbone. At the HSC pixel scale of
0.168 arcsec those scales are:

    PSF core            3.6-4.8 px   seeing 0.6-0.8 arcsec
    tightest pairs      2.2-4.5 px   measured minimum and 10th percentile
    median pair         8.9     px   theta is log-uniform on [seeing, 4"]
    widest pair         24.3    px   theta_max = min(30 proper kpc, 4")
    tidal features      60+     px   truncated by the 96 px field

Separation is drawn above the seeing but the offset is rounded to whole
pixels, so the realised minimum falls below the nominal floor. Twenty-seven
percent of the pairs sit inside 1 arcsec, which is also where the central
source's own core lives; that number sets the innermost ring boundary.

To measure a structure of size s -- not merely notice it -- a unit needs a
stride of at most s/2, so the structure is sampled at all, and a receptive
field of roughly 2s, so the unit sees the whole structure plus enough
background to normalise it against. Every scale above therefore needs a stage
that satisfies both:

    layer1  96x96   stride  1   RF  11 px   PSF core, seeing-limited pairs
    layer2  48x48   stride  2   RF  26 px   the bulk of the separation range
    layer3  24x24   stride  4   RF  56 px   widest pairs plus environment
    layer4  12x12   stride  8   RF 116 px   full field: tidal features

There is no fifth stage, but not because a theoretical receptive field of
116 px covers the 96 px cutout. Effective receptive fields grow like the
square root of depth and are a fraction of the theoretical one (Luo et al.
2016); measured on this design's predecessor, a corner unit of the final map
drew 0.7 percent of its response from the central 12.5 px. The real reason is
that only the merger class needs a scale coarser than layer4, and a residual
block costs 2*C^2*9 parameters wherever it sits, so each further stage is the
most expensive one to add for the narrowest benefit.

``layer1`` .. ``layer4`` are the unfreezing units used by transfer learning,
and ``layer3`` -- stride 4, receptive field 56 px, the scale the separation
lives on -- is the Grad-CAM target.

What the network is not given: theta_max depends on redshift through
30 proper kpc, and theta's lower bound is the seeing of that exposure, so the
class boundary is a function of (z, seeing) that varies image to image. This
model sees pixels only, so the scale table above describes the population's
marginal distribution, not any single cutout's, and the network has to infer
"resolved relative to what" from the image itself.

Those two omissions are not equally forced, and an earlier version of this
paragraph treated them as one. Redshift is unavailable for most deployment
targets, so dropping it buys reach. Seeing is available for every HSC cutout --
the simulation reads it already, since theta's lower bound is the pair's seeing
-- and it is the decisive nuisance for exactly the tightest pairs: the same
pair is resolved at 0.6 arcsec and not at 0.8, and on pixels alone those are
the same image. It stays out here because feeding it is a data question, not an
architectural one -- the cutout store carries no seeing column -- and the head
can now use it, which it could not when it was affine. When the column exists,
the change is ``forward(self, x, meta=None)`` with ``meta`` given its own
``BatchNorm1d`` and concatenated *after* ``self.drop``: a scalar folded into the
2,392 pooled statistics would be dropped 40 percent of the time, which for a
single decisive nuisance is worse than not having it. Ninety-six parameters per
scalar. It is not added ahead of the data, because an argument nothing passes is
the dead code this file has just finished removing.

Two further things the schedule above does not reach at all.

single_star against single_agn, and dual_agn against agn_star, are pairs whose
members put the same unresolved point source in the same place. The only
physical difference is the spectral energy distribution. The ring geometry, the
edge table and the eight-fold dihedral augmentation are structurally silent on
both, and no amount of work on RingPool will move them: what decides them is
the three band values and whatever the head makes of their ratios. If the
confusion matrix's mass sits there, the useful change is more bands or an
explicit photometric input, not a finer radial schedule.

And "whatever the head makes of their ratios" is doing more work in that
sentence than the normalisation supports. ``asinh_normalize`` applies
asinh(x/s)/asinh(1/s) with s = 0.1, which is logarithmic only for x >> s; there
a difference of two bands is a log flux ratio, that is, a colour. Measured on
the stored cutouts, the sky sits at x = 0.12, so x/s = 1.2, where the stretch
has 77 percent of its logarithmic slope; the 90th percentile pixel reaches 0.90
and only the 99th, x = 0.98, reaches 0.995. Hold a true colour fixed at
i/g = 2 and sweep the brightness across that range and the band difference runs
0.017, 0.064, 0.134, 0.198, 0.225, 0.230 -- a factor of fourteen, converging on
the real colour ln(2)/asinh(10) = 0.231 only above the 99th percentile. On top
of the actual sky an injected companion of contrast 0.003 to 0.30 gives 0.006,
0.020, 0.052, 0.117, 0.177: very nearly proportional to its brightness over the
whole range alpha spans.

So a band difference is a colour for the bright core of a central point source
and a brightness-colour blend everywhere else, and the faint offset companion
that separates dual_agn from agn_star is in the second regime. Separating the
two there needs a division the convolutions can only approximate piecewise,
performed where the companion is still spatially isolated -- by the time a ring
statistic is formed its light is summed with the central source's. The cheap
fix is not architectural: an explicit Lupton-style colour channel,
asinh(I/s_I) - asinh(R/s_R) at a noise-matched softening, makes the colour a
quantity the convolutions read locally at the companion, and ``channels`` is
already a constructor argument. That is a dataset change, so it is noted here
and not made. What it cannot fix is that three bands hold two colours and the
stellar and quasar loci overlap in gri; that part is a limit of the data.

And nothing here is invariant to overall brightness. ``asinh_normalize`` uses
fixed dataset-level vmin and vmax, so every ring statistic is an absolute
quantity, and there is no instance normalisation anywhere in the network. The
eight classes have quite different magnitude selection functions by
construction, which makes total flux a usable shortcut; it is suppressed
entirely on the data side -- the bright-merger attenuation, the brightness
audit -- and not at all here. That is a note on where the assumption lives, not
a complaint: it means a change of data source can invalidate it silently.
"""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

DRAGON_CUTOUT_SIZE = 96

# Three 2x stages, so every stage sees an even-sized map and samples it
# symmetrically.
DRAGON_SIZE_DIVISOR = 8

# Ring boundaries for the head, in input pixels: 0.5, 1, 2 and 4 arcsec at
# 0.168 arcsec/pixel. The inner ones bin the separation itself. Anchoring to
# angle rather than to a fraction of the map keeps the same physical bins for
# every cutout size.
#
# The 0.5 arcsec boundary separates the central source from the closest
# companion it could have; otherwise both sit in ring 0. At seeing
# 0.6-0.8 arcsec the central core reaches 0.3-0.4 arcsec and the nearest
# resolvable companion sits at the seeing itself, so a boundary between those
# two radii is fixed by the seeing alone. That matters: it does not depend on
# how the separations are distributed, and that distribution is a choice the
# simulation made, not a property of the sky. Twenty-seven percent of the
# simulated pairs do fall inside 1 arcsec, so the new boundary also carries a
# lot of them, but that is corroboration and not the reason. The edge
# survives only at stride 2, where 3 px is exactly the 1.5 map-pixel floor
# below; stride 4 and 8 drop it on their own.
#
# The 4 arcsec edge is the one constant here taken from the task's scope
# rather than from the optics. It is theta_max, so the outermost ring is
# "beyond any companion of interest" -- true of the training set by
# construction, and true of the sky only in the sense that this search is
# defined to stop at 4 arcsec. A real companion at 5-8 arcsec gets no ring of
# its own: it falls in the field ring, which keeps only a mean, so it reads
# as a brighter field and not as a companion. Widening the search means
# changing this constant, not just retraining on wider pairs.
#
# The table is canonical in whole input pixels and the angles are what those
# pixels mean, not the other way round. Deriving it the other way would divide
# 0.5 arcsec by 0.168 to get 2.976 px, whose stride-2 image 1.488 falls just
# below DRAGON_MIN_RING_EDGE_MAP_PX -- silently deleting the innermost boundary
# at the one stage that carries it.
DRAGON_RING_EDGES_PX = (3.0, 6.0, 12.0, 24.0)

# An edge closer in than this many feature-map pixels is dropped for that
# stage: the radii available on an N x N grid are 0.707, 1.581, 2.121, ...,
# so a boundary below 1.5 map pixels would leave the ring behind it empty.
# This is the stride <= s/2 rule again -- at stride 8 a 6 px boundary is
# simply not sampled.
DRAGON_MIN_RING_EDGE_MAP_PX = 1.5

# The outermost ring has to contain at least one position at every stage, so
# every stage's map corner radius sqrt(2)*(N-1)/2 must exceed that stage's
# last edge. layer4 binds: it sees one eighth of the input against an edge of
# 3 map pixels, and a 5x5 map reaches only 2.83. That puts the minimum input
# at 48; layer2 and layer3 already clear their edges at 40.
DRAGON_MIN_SIZE = 48

# Single-sourced checkpoint anchors, so a script that rewrites a saved
# state_dict does not hard-code module paths. ``__init__`` checks them against
# the built model, which is why they live here and not in the converter:
# renaming a layer then fails at construction instead of silently breaking
# conversion much later. Each is a prefix of the state_dict names it claims.
#
# The first convolution is rewritten in place when the channel count changes.
# The head is replaced wholesale whenever it no longer fits -- a different
# class count, a different ring-statistic width, a different layout -- so it is
# named by prefix, except the classifier bias, which is named exactly because
# that is where the checkpoint's class count is read from.
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
    """Anti-aliased 2x downsample: replicate-padded binomial low-pass, then
    subsample by 2.

    Low-pass before subsampling is Nyquist, and it is the whole of what this
    layer does. It deliberately does *not* take a max first, which is the
    usual pairing (Zhang 2019) and was this file's earlier design.

    The argument for a max is that average pooling divides a point source's
    peak by four while dividing its noise only by two. That holds for a source
    confined to one pixel, and the HSC PSF is not: at 3.6-4.8 px FWHM it is
    roughly twice oversampled. Measured on a 4 px FWHM Gaussian in noise, the
    peak SNR after one 2x downsample changes by

        AvgPool 1.45x   MaxPool 1.39x   BlurPool 2.23x   MaxBlurPool 1.98x

    and in the regime the max was introduced for -- a companion at five
    percent of the primary -- by 1.42x, 1.32x, 2.08x and 1.83x. Averaging a
    well-sampled PSF is a crude matched filter and *raises* SNR; the max is a
    positively biased order statistic that costs some of that back. The same
    measurement with a one-pixel source gives 0.50x for AvgPool, which is the
    case the older docstring was built on and not the case this data is in.

    The pooling acts on feature maps, not on the image, so the premise was
    checked there too: fed a 4 px FWHM point source, layer1's per-channel
    responses have a median effective FWHM of 4.4 px -- no super-resolution.
    A minority are sharper (10th percentile 2.0 px), and for those channels
    the max would still have something to offer; that is the one part of this
    trade that is not settled.

    The four-tap [1,3,3,1] kernel is what an even-sized map needs. A three-tap
    kernel samples an even map asymmetrically -- the last window stops one
    pixel short of the padded end -- which would break the exact equivalence
    of the eight dihedral training views.

    The border is replicated, for the reason ``_conv_bn`` gives. Reflection and
    replication both hold unit gain exactly and both leave the eight views
    exactly equivalent -- measured, they agree to 1.8e-7 either way -- so
    nothing here chooses between them, but reflection mirrors a source near the
    edge into a second copy of itself, which is the structure this network
    looks for. At one pixel of padding that ghost lands well outside theta_max
    and the numbers barely move; the point is that this file should not argue
    against reflection in ``_conv_bn`` and then use it here.
    """

    def __init__(self, channels):
        super().__init__()
        weights = torch.tensor([1.0, 3.0, 3.0, 1.0])
        kernel = torch.outer(weights, weights)
        kernel = kernel / kernel.sum()
        # A fixed filter, so keep it out of the checkpoint.
        self.register_buffer(
            "blur", kernel[None, None].repeat(channels, 1, 1, 1), persistent=False
        )
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.blur, stride=2, groups=self.channels)


class Downsample(nn.Module):
    """Activation, then ``BlurPool2d``.

    The blur is linear, so without this activation the path from a residual
    sum through the blur into the next stage's convolution would be two linear
    maps in a row. The activation is not in place: it is applied to the
    previous stage's output, which the autograd graph still needs.
    """

    def __init__(self, channels):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.pool = BlurPool2d(channels)

    def forward(self, x):
        return self.pool(self.act(x))


def _conv_bn(in_channels, out_channels, kernel_size=3):
    """Convolution without bias, since the BatchNorm shift subsumes it.

    Padding replicates the border rather than filling it with zeros.
    ``asinh_normalize`` clamps to ``[0, 1]`` and its stretch expands the faint
    end, so the sky sits at a clearly positive value and a zero-filled border
    is darker than any realistic pixel -- an out-of-distribution rim drawn
    around every cutout, whose kernel-mass deficit (2/3 along an edge, 4/9 in
    a corner) BatchNorm cannot undo because it rescales a whole channel at
    once. Reflection would keep unit gain too, but it mirrors a source near
    the edge into a second copy of itself, which is precisely the structure
    this network is trained to find. Replication does neither.

    It is not free: PyTorch implements any padding_mode other than ``zeros``
    as a separate ``F.pad`` kernel followed by an unpadded convolution, so it
    forgoes cuDNN's fused padding. Measured on the 96x96 24-channel layer that
    costs about 60 percent more per call, and ``layer1`` runs four of them.
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
    """``_conv_bn`` plus the activation, for the interior of a stage.

    Stage *outputs* deliberately do not end in an activation -- every consumer
    rectifies what it reads, because a residual sum is signed. ``layer1`` used
    to break that convention by ending in a LeakyReLU that ``Downsample`` then
    applied a second time, multiplying the negative tail by 0.01 twice over:
    an effective slope of 1e-4, cancelling the leak exactly where dead stem
    units are most likely and where nothing else can compensate, since
    ``layer1`` is not tapped and ``layer2`` is its only consumer.
    """
    return nn.Sequential(
        *_conv_bn(in_channels, out_channels, kernel_size),
        nn.LeakyReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions on a clean identity shortcut.

    Nothing follows the sum. Paired with the zeroed ``bn2.weight`` set in
    ``DRAGON._initialize_weights`` the block is then exactly the identity at
    init, and the shortcut carries gradients unscaled. A LeakyReLU after the
    sum would break both: the incoming tensor is already an activation, so a
    second LeakyReLU multiplies its negative tail by another 0.01, and it
    scales shortcut gradients by 0.01 wherever the sum is negative.

    One block per stage, and the design does not want more. Because no
    activation follows the sum, stacking two of these puts ``bn2`` of the first
    directly into ``conv1`` of the second with no nonlinearity between them,
    wasting a layer; a pre-activation ordering would lift that, but nothing
    here wants to stack. At thirteen learned convolutions the network is not
    deep enough for the shortcut to be carrying trainability -- BatchNorm alone
    handles that depth. What it carries is the identity initialisation, which
    starts the model as a seven-convolution plain net and lets it grow into the
    rest, and that is worth having under a cosine schedule with no warmup.

    Anything that consumes a block's output has to rectify it first --
    ``Downsample`` and the ring taps both do.
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
    """Concentric-ring pooling: per ring, the mean and the max of each channel.

    The labels are made out of a radius. The central source sits at the exact
    centre of the cutout and a companion is a companion only within theta_max
    of it, so "how far from the centre" is the quantity the classes are
    defined by. Global average pooling maps a companion at 5 px and one at
    40 px to the same number, and the second is a different object.

    ``edges`` are radii in feature-map pixels, from ``DRAGON_RING_EDGES_PX``,
    which is fixed in angle. The *inner* boundaries therefore mean the same
    thing at every stage and every cutout size; the last ring does not -- it
    runs from the outermost edge to the map corner, so its extent, and the
    statistics BatchNorm learns for it, do depend on the cutout size. An
    earlier version anchored every boundary to a fraction of the map's own
    width; at stride 4 that put the inner boundary at 3.8 arcsec, so the
    entire separation range 0.6-4 arcsec fell in one ring and no radial
    information survived at all.

    Rings stay exactly invariant under all eight dihedral transforms, because
    every ring is a dihedral-symmetric set of positions.

    Both statistics are kept, on different inputs. The max reads a rectified
    map, because ``amax`` over a signed one is a dilation that erodes whatever
    the network encoded negatively. The mean reads the raw signed map: it is a
    linear functional, so a signed input causes it no trouble, and rectifying
    first would compress a negative-coded feature -- a deficit or asymmetry at
    some radius -- by a factor of a hundred for nothing. Measured on a trained
    residual sum, the two versions of the innermost ring's mean correlate at
    only 0.86, so the difference is real.

    The outermost ring keeps only its mean unless ``outer_max`` says
    otherwise. It is the "beyond theta_max" ring, where by construction no
    companion of interest sits, so "is there a peak out there" is not a
    question worth another C dimensions; its job is to describe the field,
    which is what a mean does.

    A third reason used to stand here -- that the outer ring's mean and max are
    the most redundant, correlating at 0.96 against 0.26 innermost -- and it is
    withdrawn. Measured on the trained model's own feature maps the ordering is
    the other way round: outermost r = 0.00, 0.03, 0.19 and 0.36 at taps 1 to 4,
    innermost r = 0.71, 0.91, 0.93 and 0.94. That is what the geometry should
    have predicted -- an innermost ring of four positions has a max that is one
    of four numbers whose mean it is being compared against -- and 0.96 was
    almost certainly measured on raw cutouts, where an outer ring is mostly sky
    and both statistics track the sky level. So the outer max is not redundant
    anywhere; the case for dropping it at layer1 and layer2 is the two reasons
    above and the receptive-field rule below, and rests on nothing else.

    ``layer3`` and ``layer4`` override that. Both reasons above are reasons
    about companions, and merger is not a companion class: its evidence is a
    tidal feature at 20-60 px, most of which is a radius beyond the 4 arcsec
    edge and therefore in exactly this ring. It is localised and azimuthally
    asymmetric, so the ring's mean divides it by the ratio of the ring's
    positions to the ones it occupies, while ``max - mean`` is its detector.
    On the trained model these two rings are where the max is least redundant
    of all -- r = 0.19 and 0.36 against the mean -- so this override is not
    buying back a statistic the mean already explained.

    So the rule is not "outermost". It is whether one position of the map
    already integrates an area the size of the structure the ring has to
    report: 56 px of receptive field at layer3 and 116 px at layer4 against a
    20-60 px feature, but 26 px at layer2 and 11 px at layer1, where the same
    max would report the brightest small patch in the field instead -- and
    field sources are common to every class. Their outermost rings also hold
    1,856 and 7,412 positions, so the max there is an extreme-value statistic
    before it is anything else.

    Those are two clauses and only the first is scale-free, so it is the rule
    and the second is corroboration. Applied to the inner rings the first
    clause passes everywhere: what a 0.5-4 arcsec ring has to report is a
    companion, which is a PSF of 3.6-4.8 px, and 11 px of receptive field at
    layer1 already integrates one. The second clause does not -- tap1's
    2-4 arcsec ring holds 1,356 positions and tap2's holds 336 -- so those
    maxes are extreme-value statistics too. They are kept anyway, because there
    the extreme is the thing being looked for, and because the simulation's
    S/N >= 5 acceptance gate on the injected source bounds how far a noise peak
    can outrun it. On the outermost ring no companion of interest is present at
    all, which is what makes the same position count decisive there.

    Keeping both statistics raw also hands the head their difference, and that
    difference is the useful one. The *central* source is centred, so its light
    is azimuthally symmetric and a ring's mean is where it lands; the offset
    source is not, so it lifts that ring's max more than its mean.
    Measured on a synthetic centre-plus-companion map, a channel-space
    descriptor built from ``max - mean`` separated two companion types about
    twice as sharply as one built from the mean alone.

    Two qualifications, both of which the head has to clean up. "Central" is
    not "carrier": the simulation balances the carrier role 50/50 between the
    central and the offset source, so in half the pairs the offset source
    carries full amplitude and does move its ring's mean appreciably.
    ``max - mean`` stays an asymmetry detector there; it stops being
    proportional to the offset source's flux.

    And the argument holds only on a thin ring. These are an octave wide, so a
    centred source with any radial gradient puts its max at the ring's inner
    edge and its mean well below it, with no companion involved at all. On a
    purely azimuthally symmetric 4 px Moffat, ``(max - mean)/mean`` measured on
    the real tap geometries runs 0.00, 1.11, 1.47, 3.56, 5.51 at stride 2 and
    0.00, 1.73, 2.65 at stride 8: only the innermost ring, whose four positions
    share one radius, is clean. So ``max - mean`` is the radial gradient plus
    the asymmetry, and separating them means dividing by the mean.
    ``feature_norm`` is affine per feature, so any linear combination of these
    outputs is reachable; that ratio is not, which is what the head's hidden
    layer is for. Narrowing the rings would attack it directly, but the edge
    table is the label geometry and a boundary at the geometric midpoint of
    2-4 arcsec, 2.85 arcsec, would be a numerical convenience with no
    counterpart in the labels.

    The matched statistic for that contamination exists and is not here, which
    needs saying because the reason is not the obvious one. An azimuthal dipole
    modulus, |sum_ring x e^{i phi}| / sum_ring |x|, is exactly zero on any
    azimuthally symmetric light distribution however steep its radial profile
    -- measured, 1e-17 -- is exactly invariant under all eight dihedral
    transforms, is bounded in [0, 1] with no denominator that can change sign,
    and is a modulus, so unlike ``max - mean`` no affine head can build it. On
    raw cutouts it is worth a great deal: for a 5 percent tidal arc against a
    varying Sersic profile and a factor-of-ten brightness nuisance it lifts the
    best linear combination's Fisher d from 1.3 to 2.7, and for a companion in
    the separation rings from 1.0-2.0 to 2.6-3.8.

    It is not here because the network already has it. A rectified sum over K
    evenly spaced oriented channels reconstructs a modulus to 6 percent ripple
    at K = 8 and 2 percent at K = 16, layer3 has 128 channels, and eight-fold
    dihedral augmentation is what makes the stack learn the orientations. So
    the question is whether the information survives to the head, and measured
    on the trained model it does: adding the dipole to the pooled statistics,
    at its own natural scope or at every max ring, moves held-out balanced
    accuracy from 0.890 to 0.886 and 0.882 -- nothing, minus the cost of 224 or
    1,152 extra dimensions. Nor is merger short of evidence, which was the
    premise: a linear probe on the pooled statistics alone separates it from
    single_galaxy at AUC 0.995 and from agn_galaxy at 0.998. The contamination
    above is real; it is not what limits this network.

    Masks depend only on ``(height, width)`` and are built once per size. The
    emptiness check happens then, never in the hot path: testing a CUDA
    boolean tensor forces a device-to-host synchronisation, which would break
    the stream on every forward.
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
        # Which rings contribute a max is fixed by ring index, not by how many
        # positions a ring happens to hold, so the output width does not depend
        # on the input size.
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
    """Downsample, project to the stage width, then one residual block.

    ``project`` is not always a widening: layer4 narrows, because a
    same-width 3x3 there bought nothing but 17 percent of the model's
    parameters. Not because it serves a single class -- a third of the pairs
    sit beyond 2 arcsec, inside layer4's second ring, and its outer ring is
    the field descriptor every class reads. The defensible claim is narrower:
    coarse-scale channel semantics take fewer channels to describe than
    fine-scale ones.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = Downsample(in_channels)
        self.project = _conv_bn_act(in_channels, out_channels)
        self.block = ResidualBlock(out_channels)

    def forward(self, x):
        return self.block(self.project(self.down(x)))


class RingTap(nn.Module):
    """Rectify a stage's output, then ring-pool it.

    The activation is not optional. A stage returns a ``ResidualBlock`` sum,
    which carries no activation by design and is signed. Measured on the
    trained model over 960 devel cutouts, 26 to 32 percent of a residual sum's
    entries are negative, carrying about 10 percent of its absolute magnitude;
    an earlier version of this docstring said 35 percent of the magnitude,
    which was too high, and the technical report said 50, which is layer1's
    figure -- layer1 ends at a BatchNorm with no residual block at all.
    ``amax`` over a signed map is still a dilation: it keeps peaks and erodes
    troughs, deleting whatever the network chose to encode negatively. Ten
    percent of the magnitude is what that deletion is worth here, not half.

    At initialisation the residual branch is exactly zero, so a stage's output
    is exactly the projection's post-LeakyReLU tensor -- verified, the two
    agree bit for bit -- of which 49 percent is already the leak. The rectifier
    below then multiplies that leak by 0.01 a second time, the 1e-4 slope
    ``_conv_bn_act`` names as a bug, in layer2, layer3 and layer4 at once. It
    is transient, and the negative tail it squashes is small at init because
    the projection has already scaled it by 0.01; but the invariant "a stage
    output is signed, so its consumer must rectify" is false at exactly the
    step a warmup-free cosine schedule starts from, and that is worth knowing
    when reading the first epoch's behaviour.

    Only the max branch is fed the rectified map. The mean is linear and takes
    the stage's output as it stands -- see ``RingPool``.
    """

    def __init__(self, edges, *, outer_max=False):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.pool = RingPool(edges, outer_max=outer_max)

    def out_features(self, channels):
        return self.pool.out_features(channels)

    def forward(self, x):
        return self.pool(x, self.act(x))


class DRAGON(nn.Module):
    """Four-stage CNN for close-pair classification in survey cutouts.

    ``channels`` and ``num_classes`` are required: they must agree with the
    stored cutouts and with ``labels.csv``, and a default that silently
    disagreed with either would surface only as a shape error much later.

    ``widths`` is the only capacity knob on the backbone, and
    multiply-accumulates and parameters must not be conflated. Moving capacity
    towards the fine scales costs arithmetic quadratically, because the map is
    sixteen times larger at stride 1 than at stride 4; moving it towards the
    coarse ones costs
    parameters, because a residual block costs 2*C^2*9 wherever it is placed
    and the widest stage is always the last. Wall-clock follows the
    arithmetic.

    What the parameters buy is not, on this data, protection from i.i.d.
    overfitting: 987k parameters against 480k training cutouts under an exact
    eight-fold dihedral augmentation is a small model, and wall-clock binds
    first. Two other things do scale with capacity. Three of the eight classes
    exist only as simulations, so whatever the model learns about how those
    images were assembled transfers to nothing at deployment, and a wider
    model learns more of it. And the sample size that matters is per class,
    not in total: merger has 8,751 training images, and it is the class
    layer4's coarse scales chiefly serve.

    It is tempting to localise that first risk -- to a stride, or to which
    stages the head can see -- and this file did, wrongly, for a while. The
    correction is what settles the layer1 tap, so it is worth stating.

    The shortcut is real: Y = C + alpha*M*S with no background subtraction
    raises the noise variance inside the injected source's mask support by
    1 + alpha^2. It is not confined to fine strides, because a stage rectifies
    before it pools. A high-pass filter followed by a LeakyReLU turns noise
    power into a non-negative map, and BlurPool is a unit-gain low-pass, so it
    carries that map's local mean through untouched. Measured on the real
    BlurPool chain against a 4.4 percent sigma step, Cohen's d is 0.093, 0.095,
    0.097 and 0.097 at strides 1, 2, 4 and 8 -- no attenuation at any depth.
    Nor is it confined to a tapped stage: layer2 can carry layer1's rectified
    noise-power channel and tap2 pools it like any other.

    It is also weak, but the bound has to be stated as a bound and not as a
    single feature's effect size. alpha is uniform on [0.05, 0.30), so the
    sigma step runs from 0.12 to 4.4 percent, and a sample standard deviation
    over N pixels has relative precision 1/sqrt(2N). The 0.09 above is what one
    pooled statistic reads; the head sees 2,392 of them over 312 channels, and
    weak cues that are read repeatedly add in quadrature, so the number that
    matters is the ceiling over every estimator. That ceiling is
    d <= 0.044*sqrt(2N) -- about 0.9 at N = 200 pixels of mask support and the
    largest alpha, about 0.3 at the median one -- and it is a ceiling because
    the 312 channels are 312 looks at one noise field, not 312 independent
    cues; no amount of width buys information the pixels do not hold. So the
    per-image discriminant is bounded near d ~ 0.3 while a per-population cue
    is not bounded at all, and the population is large: 204,039 of the 480,278
    training images, 42 percent, are simulated -- 70,000 dual_agn, 70,000
    agn_star and 64,039 agn_galaxy. An earlier version of this docstring said
    7,000 per class, ten times low.

    So nothing in the architecture blocks it and nothing here is trying to.
    Total capacity bears on it, which is the paragraph above; the data pipeline
    bears on it; no stage's parameters are "safe" on this ground.

    The default trades some of the second for some of the first relative to a
    naively monotonic width table: layer2 is widened and layer4 narrowed,
    which moves parameters from a stage serving one class to one covering the
    separation range, for about 13 percent more arithmetic. It does not go
    further than that, and by this file's own stride <= s/2 rule it arguably
    should: layer3 can only measure structure of 8 px and up, which is the
    upper half of the separation distribution, and it holds 37 percent of the
    parameters, while layer2 -- the only stage that measures 4-8 px and the
    only one that keeps the 0.5 arcsec ring boundary -- holds 9. Those two
    stages also serve the class with the fewest training images, which the
    balanced loss weights by 6.9.

    The obvious alternative is (24, 96, 96, 64): 803k parameters against 987k,
    754 MMAC against 616, layer2 at 23 percent and layer4 at 16. It is not
    free -- the map is four times larger at layer2, which is the whole of the
    extra arithmetic -- and the rule it leans on says which stage can *measure*
    a separation, not how many channels companion typing needs, which is a
    different question and the one layer3's width is actually answering. So the
    table is left where measurement can decide it.

    layer4 is the stage with the weakest case, and for a reason the width rule
    does not state. Its theoretical receptive field, 116 px, exceeds the 96 px
    field, and only two ring edges survive its stride, so tap4 reports three
    rings holding 4, 28 and 112 map positions. Those are close to global
    descriptors, and there are not many global configurations eight classes can
    differ by. Narrowing it -- (24, 64, 128, 48) -- costs 21 percent of the
    parameters and 4 percent of the arithmetic, but it concentrates rather than
    rebalances: layer3 then holds 47 percent. That is the argument this table
    is waiting on measurement for.

    ``head_width`` sizes the head's one hidden layer. The head has to be more
    than affine because three of the eight classes are separated by colour and
    not by geometry: dual_agn and agn_star both put an unresolved point source
    at the same radius, and single_star and single_agn both put one at the
    centre. Colour is a ratio of band fluxes, an affine map cannot form a
    ratio, and the nuisance it would have to divide out is real -- alpha spans
    a factor of six and scales the offset source's contribution to a ring
    statistic without scaling the central source's.

    For the binary question, is there a companion, that costs only
    calibration: alpha is positive and a positive rescaling leaves the sign of
    a linear discriminant alone. That much is true, and it is the whole of the
    case for an affine head. It does not reach the three-way question, which
    companion, where the logit difference is
    (w_i - w_j).a + alpha*(w_i - w_j).b with a the central source's
    contribution -- label-irrelevant, and varying image to image independently
    of alpha. One hyperplane then has to separate a family whose
    signal-to-nuisance ratio moves by a factor of six, and that costs sight,
    not calibration. The same gap is why "a per-ring normalisation changes
    nothing a cosine can see" does not settle that question either: it is true
    of the feature direction and irrelevant to a classifier that reads the
    unnormalised vector.

    A second ratio is needed whether or not colour is. ``max - mean`` is the
    asymmetric part of a ring only on a thin ring, and these are an octave
    wide; on a purely symmetric centred source it is dominated by the radial
    gradient (see ``RingPool``). Recovering the asymmetry means dividing by the
    mean, which is again something no affine head can do.

    The width does not follow from a rank count, and an earlier version of
    this docstring tried to make it: 8 for the discriminant, 32 for the
    (ring, statistic) slots, 15 for the ratios, so 96. The last term is a
    category error. A rank budget counts linear directions and a quotient is
    not one; a shallow LeakyReLU net approximating a smooth 1-D function to
    accuracy epsilon needs order epsilon^-1/2 breakpoints, and a/b in both
    arguments is worse. "One unit per ratio" buys nothing. The honest statement
    is that 96 is a guess: the head has to be non-affine for the reasons above,
    nothing here derives how non-affine, and 64 was raised only because tap1
    and layer3's outer max widened the input under it. The hidden layer is
    followed by BatchNorm, so its initialisation scale does not matter.

    Two coordinate changes that would make the head's job easier were tried on
    the trained model and both lost. Replacing each max by the dimensionless
    (max - mean_+)/(mean_+ + eps), so the ratio the head is being asked for
    comes free from the pooling layer, costs 1.4 points of held-out balanced
    accuracy on a linear probe of the pooled statistics (0.890 to 0.876) and is
    worse on every class boundary tested; adding it alongside the max rather
    than in place of it recovers nothing. It is also unsafe: the denominator is
    a rectified ring mean, and LeakyReLU leaves that negative on 3 to 10 percent
    of (channel, ring) slots on the trained model and on 20 to 46 percent at
    initialisation, where a stage output is 49 percent leak. See the
    azimuthal-dipole note in ``RingPool`` for the second.

    What survives that critique is the objection to where the dropout sits.
    p = 0.4 on the 2,392 pooled statistics leaves any particular (mean, max)
    pair jointly present in 36 percent of steps, and a ratio is a co-adaptation
    of exactly that pair -- the thing dropout exists to suppress. That is a real
    tension with the paragraph above and it is not resolved here; it is stated
    because the alternative, moving the dropout onto a 96-unit bottleneck,
    trades it for a worse problem, and because the measurement that would
    settle it is a training run.

    The dropout sits on the 2,392 pooled statistics rather than on the hidden
    layer. Its rate was inherited from an affine head, where it thinned a wide
    redundant vector; at p = 0.4 a 96-unit bottleneck keeps 58 units on
    average, close enough to the count above to risk starving the rank it was
    just sized for. The wide side is also where the parameters are -- the
    hidden Linear is 98 percent of the head. The cost is a BatchNorm
    downstream of a dropout, whose train-time variance inflation leaves the
    running variance an overestimate; that lands as a near-uniform rescaling
    of the hidden units, which the decision-threshold calibration absorbs,
    where a starved bottleneck would not be absorbed by anything.

    Two limitations are left in place deliberately. The width table is left
    for measurement, above. And the model is given no seeing and no invariance
    to overall brightness -- see the module docstring for both.
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

        # Four convolutions at stride 1 take the receptive field to 11 px, so
        # a unit finally sees a whole PSF and both members of a seeing-limited
        # pair before anything is thrown away. Narrow on purpose: within
        # 11 px the light distribution is described to second order by of
        # order ten numbers per band, and the map here is sixteen times larger
        # than at stride 4.
        # The last block stops at the BatchNorm: every consumer of a stage
        # output rectifies it, and layer2's Downsample is this one's only
        # consumer.
        self.layer1 = nn.Sequential(
            _conv_bn_act(channels, stem, kernel_size=5),
            _conv_bn_act(stem, stem),
            _conv_bn_act(stem, stem),
            nn.Sequential(*_conv_bn(stem, stem)),
        )
        self.layer2 = Stage(stem, first)
        self.layer3 = Stage(first, second)
        self.layer4 = Stage(second, third)

        # Four taps. The reason for reading below the last stage is that
        # pooling and nonlinearity do not preserve a quantitative "how far
        # apart", and that argument does not stop at layer3: by this file's
        # own stride <= s/2 rule, stride 4 can only measure structure of 8 px
        # and up, while the tightest pairs are 2.2-4.5 px and are resolved only
        # at layer2 and finer. layer2 (RF 26 px) covers the separation range,
        # layer3 (RF 56 px) covers it with environment, layer4 covers the
        # field the merger class needs.
        #
        # The same rule reaches layer1, which is why it is tapped. The four-tap
        # [1,3,3,1] kernel has transfer function (cos(1.5w) + 3cos(0.5w))/4,
        # which passes 0.83 at the median separation of 8.9 px, 0.64 at
        # 1 arcsec, 0.43 at the 10th-percentile 4.4 px, 0.27 at 3.6 px, and
        # nothing at all at the realised minimum 2.2 px, past Nyquist for a
        # stride of 2. So layer1 is the only stage that sees the tightest
        # quarter of the pairs at full contrast, and the only one where central
        # compactness is resolved -- which is what separates merger from
        # agn_galaxy, since both put extended light off centre and only one
        # puts a point source at the middle. It costs 24*9 = 216 features;
        # recompute that if DRAGON_RING_EDGES_PX changes, since it is the
        # four-edge table that gives 5 rings and 4 maxes at stride 1.
        #
        # It was withheld for a while on the grounds that tapping the one
        # stride-1 stage would hand the head the simulation's own noise
        # signature. That argument does not survive measurement -- the
        # signature passes every BlurPool undiminished and reaches the head
        # through tap2 regardless -- and the class docstring has the numbers.
        stage_strides = (1, 2, 4, 8)
        edges = [_stage_ring_edges(stride) for stride in stage_strides]
        self.tap1 = RingTap(edges[0])
        self.tap2 = RingTap(edges[1])
        # The outermost ring is the one that has to report a 20-60 px tidal
        # feature, so it keeps a max wherever one map position already
        # integrates a comparable area: 56 px of receptive field at layer3 and
        # 116 px at layer4, against 26 px at layer2 and 11 px at layer1, where
        # the same max would read the brightest small patch in the field
        # instead. See ``RingPool``.
        self.tap3 = RingTap(edges[2], outer_max=True)
        self.tap4 = RingTap(edges[3], outer_max=True)

        taps = (self.tap1, self.tap2, self.tap3, self.tap4)
        feature_dim = sum(
            tap.out_features(width)
            for tap, width in zip(taps, (stem, first, second, third))
        )
        # The mean and max branches have different scales and different
        # variances, and one linear layer under one dropout rate has to treat
        # them alike. Normalising first fixes the conditioning.
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
            # These entries carry no device, so no forward would ever hit them.
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

        # The small-std rule above is there to start the logits near zero, so
        # it belongs to the output layer only. The hidden layer is followed by
        # BatchNorm, which absorbs any init scale, and precedes a LeakyReLU, so
        # it takes the same Kaiming rule as the convolutions.
        nn.init.kaiming_normal_(self.mixer[0].weight, a=0.01, nonlinearity="leaky_relu")
        nn.init.zeros_(self.mixer[0].bias)

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
