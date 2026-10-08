# Mirror precision

A mirror reflects selected Gaussian centers about their centroid in model space.
It also changes the appropriate quaternion and spherical-harmonic signs. The
centroid uses double-precision accumulation and division, then rounds once to
float32. The CUDA reduction uses at most 8 KiB of partial sums and reads back
three floats. This prevents ordinary float32 reduction error from repeatedly
moving the pivot when mirror pairs are applied.

Two mirror commands are not a bit-exact inverse for arbitrary float32 positions.
For example, about a pivot of 1, both 0 and -2^-25 reflect to the same float32
value, 2. Reflecting that value again gives 0. The first reflection has already
lost the information needed to distinguish the original positions. Changing
node transforms cannot preserve the existing behavior for arbitrary partial
Gaussian selections without repartitioning the node or retaining extra state.

The regression contract for repeated pairs is a bound of two float32 ULPs at the
original coordinate scale, over 128 pairs on deterministic finite scenes with
full and partial selections, all three axes, and recomputed centroids. One ULP
here is `nextafter(max(abs(original coordinates)), +infinity) - scale`; it is not
an ULP of a coordinate arbitrarily close to zero. The tests also require unselected coordinates and coordinates on the other axes
to remain bit-exact, and protect exactly representable mirror pairs.

This tested bound is not a universal guarantee for overflow, subnormal ranges,
or every possible ill-conditioned point distribution. Exact restoration requires
saved pre-mirror values; a second arithmetic reflection cannot reconstruct bits
that were rounded away.
