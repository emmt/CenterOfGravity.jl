using CenterOfGravity
using Test, Random

@testset "Windowed arrays  " begin
    # Utilities.
    let to_type = CenterOfGravity.WindowedArrays.to_type
        @test to_type(Int, UInt(1)) isa Int
        @test to_type(Int, 1) isa Int
    end

    # Fully specified ranges.
    A = reshape(collect(1:20), (4,5))
    J = (2:3, 3:5)
    j_first = map(first, J)
    j_last = map(last, J)
    B = WindowedArray(A, J...)
    @test IndexStyle(B) === IndexCartesian()
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    @test axes(B) == J
    @test size(B) == map(length, J)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == [A[j1,j2] for j1 in J[1], j2 in J[2]]
    v_first = A[j_first...]
    v_last = A[j_last...]
    B[j_first...] = v_last
    @test A[j_first...] == v_last
    B[j_last...] = v_first
    @test A[j_last...] == v_first

    # Check range conversion and full range specified by a colon.
    B = WindowedArray(A, Int16(2):Int16(3), :)
    @test IndexStyle(B) === IndexCartesian()
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    J = axes(B)
    @test J == (2:3, axes(A,2))
    @test size(B) == map(length, J)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == [A[j1,j2] for j1 in J[1], j2 in J[2]]

end

@testset "Sliding windows  " begin
    # Utilities.
    let to_type = CenterOfGravity.SlidingWindows.to_type
        @test to_type(Int, UInt(1)) isa Int
        @test to_type(Int, 1) isa Int
    end
    let to_sliding_axis = CenterOfGravity.SlidingWindows.to_sliding_axis
        @test to_sliding_axis(Base.OneTo(5)) isa Base.OneTo{Int}
        @test to_sliding_axis(Base.OneTo(Int16(5))) isa Base.OneTo{Int}
        @test to_sliding_axis(-1:11) isa UnitRange{Int}
        @test to_sliding_axis(-Int16(1):Int16(11)) isa UnitRange{Int}
        @test to_sliding_axis(4) == -2:1
        @test to_sliding_axis(5) == -2:2
    end
    let to_dimension = CenterOfGravity.SlidingWindows.to_dimension
        @test to_dimension(1) === Int(1)
        @test to_dimension(Int16(1)) === Int(1)
        @test to_dimension(Base.OneTo(Int16(5))) === Int(5)
        @test to_dimension(-1:8) === Int(10)
        @test to_dimension(-Int16(1):Int16(11)) === Int(13)
    end

    # Sliding window from apprimately centered array.
    A = reshape(collect(1:20), (4,5))
    B = SlidingWindow(A)
    J = map(n -> -div(n,2) : n-1-div(n,2), size(A))
    j = map(last, J)
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    @test axes(B) == J
    @test size(B) == size(A)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == A
    val = B[j...]
    B[j...] = -val
    @test A[end,end] == -val

    # Sliding window with given axes.
    J = map(n -> Int16(0) : Int16(n-1), size(A))
    B = SlidingWindow(A, J...)
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    @test axes(B) == J
    @test size(B) == size(A)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == A

    # Uniformly true mask.
    J = (-1:7, Int16(0):Int16(4), 3:8)
    j = map(first, J)
    B = SlidingWindow(J...)
    @test eltype(B) === Bool
    @test parent(B) === CenterOfGravity.SlidingWindows.UniformlyTrue()
    @test axes(B) == J
    @test size(B) == map(length, J)
    @test [B[j1,j2,j3] for j1 in J[1], j2 in J[2], j3 in J[3]] == ones(Bool, size(B))
    @test B[j...] == true
    @test_throws ErrorException B[j...] = false
end

distance(a::NTuple{N,Real}, b::NTuple{N,Real}) where {N} =
    sqrt(sum(map((ca,cb) -> (ca - cb)^2, a, b)))

maxabsdif(a, b) = maximum(abs.(a .- b))

@testset "Center of gravity" begin
    # Utilities.
    let to_type = CenterOfGravity.to_type
        @test to_type(Int, UInt(1)) isa Int
        @test to_type(Int, 1) isa Int
    end
    let to_position = CenterOfGravity.to_position
        x1, x2 = 2.1, -1.4
        @test to_position(typeof(x1), (x1,x2)) === (x1,x2)
        @test to_position(Float64, (x1,Float32(x2))) isa Tuple{Float64,Float64}
        @test to_position(Int, CartesianIndex(3,4)) === (3,4)
    end
    let to_algorithm = CenterOfGravity.to_algorithm
        @test to_algorithm(:somemethod) === Val(:somemethod)
        @test to_algorithm(Val(:somemethod)) === Val(:somemethod)
    end

    # Model is a 2-D Gaussian.
    dims = (25,30);
    c = (9.01, 21.67); # central position
    fwhm = 4.3; # full width at half max.
    β = 4*log(2)/fwhm^2; # β = 1/2σ² with σ = fwhm/sqrt(8*log(2))
    mdl = [20*exp(-β*((x1 - c[1])^2 + (x2 - c[2])^2))
           for x1 in 1:dims[1], x2 in 1:dims[2]];
    # Add some noise.
    rng = MersenneTwister(1234); # for reproducible results
    σ = sqrt.(0.01*mdl .+ 0.3^2); # standard deviation of noise
    dat = mdl + σ .* randn(rng, Float64, size(mdl));
    wgt = 1 ./ σ.^2; # precision matrix
    # Compute center of gravity with uniformly true mask.
    c0 = map(n -> (1 + n)/2, dims); # initial position
    J = (-7:7, -7:7);

    # Tests with a uniformly true sliding window.
    win = SlidingWindow(J...);
    for r in (1 => (9.039929, 21.050560),
              2 => (8.957942, 21.617320),
              3 => (8.941763, 21.639933))
        @test maxabsdif(center_of_gravity(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.039929, 21.050560, 0.152842, -0.157663, 0.246084),
              2 => (8.957942, 21.617320, 0.023743, -0.000187, 0.025506),
              3 => (8.941763, 21.639933, 0.023928,  0.000032, 0.024067))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.039929, 21.050560, 0.014310, -0.014824, 0.023063),
              2 => (8.957942, 21.617320, 0.002216, -0.000018, 0.002381),
              3 => (8.941763, 21.639933, 0.002233,  0.000003, 0.002245))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Idem but omitting negative pixels.
    for r in (1 => (9.339042, 20.753295),
              2 => (8.974022, 21.604980),
              3 => (8.961886, 21.677506))
        @test maxabsdif(center_of_gravity(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.339042, 20.753295, 0.060124, -0.066475, 0.120475),
              2 => (8.974022, 21.604980, 0.013091,  0.000022, 0.016631),
              3 => (8.961886, 21.677506, 0.012774, -0.000589, 0.013819))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.339042, 20.753295, 0.005790, -0.006414, 0.011508),
              2 => (8.974022, 21.604980, 0.001254, -0.000005, 0.001583),
              3 => (8.961886, 21.677506, 0.001222, -0.000053, 0.001317))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Compute center of gravity with uniform weights.
    win = SlidingWindow(ones(Float64, map(length, J)), J...);
    for r in (1 => (9.039929, 21.050560),
              2 => (8.957942, 21.617320),
              3 => (8.941763, 21.639933))
        @test maxabsdif(center_of_gravity(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.039929, 21.050560, 0.107949, -0.086415, 0.098660),
              2 => (8.957942, 21.617320, 0.022160, -0.000101, 0.019831),
              3 => (8.941763, 21.639933, 0.023928,  0.000032, 0.024067))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.039929, 21.050560, 0.010270, -0.008412, 0.009795),
              2 => (8.957942, 21.617320, 0.002074, -0.000011, 0.001870),
              3 => (8.941763, 21.639933, 0.002233,  0.000003, 0.002245))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Idem but omitting negative pixels.
    for r in (1 => (9.339042, 20.753295),
              2 => (8.974022, 21.604980),
              3 => (8.961886, 21.677506))
        @test maxabsdif(center_of_gravity(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.339042, 20.753295, 0.040542, -0.035490, 0.049035),
              2 => (8.974022, 21.604980, 0.012055, -0.000072, 0.012830),
              3 => (8.961886, 21.677506, 0.012774, -0.000589, 0.013819))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.339042, 20.753295, 0.004028, -0.003626, 0.005078),
              2 => (8.974022, 21.604980, 0.001160, -0.000013, 0.001241),
              3 => (8.961886, 21.677506, 0.001222, -0.000053, 0.001317))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    ## Compute center of gravity with uniform weights without corners.
    msk = [round(Int, sqrt(x1^2 + x2^2)) ≤ 7 for x1 in J[1], x2 in J[2]];
    win = SlidingWindow(msk);
    for r in (1 => (9.669801, 20.649462),
              2 => (8.930564, 21.634711),
              3 => (8.976497, 21.650224))
        @test maxabsdif(center_of_gravity(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    # FIXME: for r in (1 => (9.669801, 20.649462, 0.145952, -0.122823, 0.179676),
    # FIXME:           2 => (8.930564, 21.634711, 0.015880, -0.001208, 0.013889),
    # FIXME:           3 => (8.976497, 21.650224, 0.014105, -0.000008, 0.014217))
    # FIXME:     @test maxabsdif(center_of_gravity_with_covariance(
    # FIXME:         dat, win, c0...; maxiter=r.first, precision=nothing),
    # FIXME:                     r.second) ≤ 1e-6
    # FIXME: end
    # FIXME: for r in (1 => (9.669801, 20.649462, 0.013782, -0.011839, 0.017608),
    # FIXME:           2 => (8.930564, 21.634711, 0.001525, -0.000127, 0.001351),
    # FIXME:           3 => (8.976497, 21.650224, 0.001349, -0.000001, 0.001359))
    # FIXME:     @test maxabsdif(center_of_gravity_with_covariance(
    # FIXME:         dat, win, c0...; maxiter=r.first, precision=wgt),
    # FIXME:                     r.second) ≤ 1e-6
    # FIXME: end

    # Idem but omitting negative pixels.
    for r in (1 => (9.943275, 20.305350),
              2 => (9.013273, 21.569760),
              3 => (8.978609, 21.663092))
        @test maxabsdif(center_of_gravity(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    # FIXME: for r in (1 => (9.943275, 20.305350, 0.060618, -0.056739, 0.104852),
    # FIXME:           2 => (9.013273, 21.569760, 0.009997, -0.001412, 0.011059),
    # FIXME:           3 => (8.978609, 21.663092, 0.008630,  0.000421, 0.009745))
    # FIXME:     @test maxabsdif(center_of_gravity_with_covariance(
    # FIXME:         :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
    # FIXME:                     r.second) ≤ 1e-6
    # FIXME: end
    # FIXME: for r in (1 => (9.943275, 20.305350, 0.005875, -0.005579, 0.010391),
    # FIXME:           2 => (9.013273, 21.569760, 0.000993, -0.000151, 0.001101),
    # FIXME:           3 => (8.978609, 21.663092, 0.000850,  0.000038, 0.000951))
    # FIXME:     @test maxabsdif(center_of_gravity_with_covariance(
    # FIXME:         :nonnegative, dat, win, c0...; maxiter=r.first, precision=wgt),
    # FIXME:                     r.second) ≤ 1e-6
    # FIXME: end

end

# for i in 1:3; @printf("%d => (%.6f, %.6f, %.6f, %.6f, %.6f),\n", i, center_of_gravity_with_covariance(:nonnegative, dat, win, c0...; maxiter=i, precision=wgt)...); end
