module BenchmarkingCenterOfGravity

using BenchmarkTools
using Random
using CenterOfGravity

# Model is a 2-D Gaussian.
dims = (25,30);
c = (9.01, 21.67); # central position
w = 4.3 # full width at half max.
σ = w/sqrt(8*log(2));
mdl = [20*exp(-((x1 - c[1])^2 + (x2 - c[2])^2)/(2*σ^2))
       for x1 in 1:dims[1], x2 in 1:dims[2]];

# Add some noise.
rng = MersenneTwister(1234); # for reproducible results
dat = mdl + 0.3.*randn(rng, Float64, size(mdl));
# Compute center of gravity with uniformly true mask.
c0 = map(n -> (1 + n)/2, dims); # initial position
J = (-7:7, -7:7);
win = SlidingWindow(J...);

for alg in (Val(:simple), Val(:nonnegative))
    print("center_of_gravity($alg, ...):")
    @btime center_of_gravity($alg, $dat, $win, $c0; maxiter=1)
    print("center_of_gravity_with_covariance($alg, ...):")
    @btime center_of_gravity_with_covariance($alg, $dat, $win, $c0; maxiter=1)
end

end # module
