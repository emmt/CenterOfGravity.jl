using CenterOfGravity
using Test, Random

include("PortableRandomGenerators.jl")
using .PortableRandomGenerators: SimpleRandomGenerator

@testset "Windowed arrays  " begin
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
    rng = SimpleRandomGenerator(0); # for reproducible results
    σ = sqrt.(0.01*mdl .+ 0.3^2); # standard deviation of noise
    dat = mdl + σ .* randn(rng, Float64, size(mdl));
    wgt = 1 ./ σ.^2; # precision matrix
    # Compute center of gravity with uniformly true mask.
    c0 = map(n -> (1 + n)/2, dims); # initial position
    J = (-7:7, -7:7);

    # Tests with a uniformly true sliding window.
    win = SlidingWindow(J...);
    for r in (1 => (9.232953, 21.075790),
              2 => (9.030415, 21.653912),
              3 => (9.018779, 21.708585))
        @test maxabsdif(center_of_gravity(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.232953, 21.075790, 0.138879, -0.147755, 0.243262),
              2 => (9.030415, 21.653912, 0.023298, -0.000264, 0.025152),
              3 => (9.018779, 21.708585, 0.023363, -0.000002, 0.023433))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.232953, 21.075790, 0.012959, -0.013866, 0.022807),
              2 => (9.030415, 21.653912, 0.002175, -0.000026, 0.002348),
              3 => (9.018779, 21.708585, 0.002180, -0.000000, 0.002187))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Idem but omitting negative pixels.
    for r in (1 => (9.444462, 20.756587),
              2 => (9.023299, 21.604873),
              3 => (9.024741, 21.698230))
        @test maxabsdif(center_of_gravity(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.444462, 20.756587, 0.071453, -0.070348, 0.115029),
              2 => (9.023299, 21.604873, 0.014904, -0.000189, 0.015715),
              3 => (9.024741, 21.698230, 0.015039,  0.000212, 0.014505))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.444462, 20.756587, 0.006770, -0.006731, 0.011007),
              2 => (9.023299, 21.604873, 0.001417, -0.000023, 0.001499),
              3 => (9.024741, 21.698230, 0.001425,  0.000020, 0.001377))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Compute center of gravity with uniform weights.
    win = SlidingWindow(ones(Float64, map(length, J)), J...);
    for r in (1 => (9.232953, 21.075790),
              2 => (9.030415, 21.653912),
              3 => (9.018779, 21.708585))
        @test maxabsdif(center_of_gravity(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.232953, 21.075790, 0.138879, -0.147755, 0.243262),
              2 => (9.030415, 21.653912, 0.023298, -0.000264, 0.025152),
              3 => (9.018779, 21.708585, 0.023363, -0.000002, 0.023433))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.232953, 21.075790, 0.012959, -0.013866, 0.022807),
              2 => (9.030415, 21.653912, 0.002175, -0.000026, 0.002348),
              3 => (9.018779, 21.708585, 0.002180, -0.000000, 0.002187))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Idem but omitting negative pixels.
    for r in (1 => (9.444462, 20.756587),
              2 => (9.023299, 21.604873),
              3 => (9.024741, 21.698230))
        @test maxabsdif(center_of_gravity(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.444462, 20.756587, 0.071453, -0.070348, 0.115029),
              2 => (9.023299, 21.604873, 0.014904, -0.000189, 0.015715),
              3 => (9.024741, 21.698230, 0.015039,  0.000212, 0.014505))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.444462, 20.756587, 0.006770, -0.006731, 0.011007),
              2 => (9.023299, 21.604873, 0.001417, -0.000023, 0.001499),
              3 => (9.024741, 21.698230, 0.001425,  0.000020, 0.001377))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    ## Compute center of gravity with uniform weights without corners.
    msk = [round(Int, sqrt(x1^2 + x2^2)) ≤ 7 for x1 in J[1], x2 in J[2]];
    win = SlidingWindow(msk);
    for r in (1 => (9.800942, 20.590818),
              2 => (9.054198, 21.631061),
              3 => (9.022643, 21.665028))
        @test maxabsdif(center_of_gravity(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.800942, 20.590818, 0.165840, -0.186804, 0.325062),
              2 => (9.054198, 21.631061, 0.016562, -0.002769, 0.016492),
              3 => (9.022643, 21.665028, 0.013900,  0.000003, 0.013989))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (9.800942, 20.590818, 0.015417, -0.017425, 0.030356),
              2 => (9.054198, 21.631061, 0.001579, -0.000265, 0.001585),
              3 => (9.022643, 21.665028, 0.001329,  0.000000, 0.001337))
        @test maxabsdif(center_of_gravity_with_covariance(
            dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

    # Idem but omitting negative pixels.
    for r in (1 => (10.007679, 20.275445),
              2 => ( 9.068533, 21.572627),
              3 => ( 9.016294, 21.676906))
        @test maxabsdif(center_of_gravity(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (10.007679, 20.275445, 0.082834, -0.085736, 0.150969),
              2 => ( 9.068533, 21.572627, 0.012358, -0.003024, 0.013808),
              3 => ( 9.016294, 21.676906, 0.010154,  0.000091, 0.009271))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=nothing),
                        r.second) ≤ 1e-6
    end
    for r in (1 => (10.007679, 20.275445, 0.007803, -0.008123, 0.014350),
              2 => ( 9.068533, 21.572627, 0.001203, -0.000295, 0.001348),
              3 => ( 9.016294, 21.676906, 0.000988,  0.000009, 0.000908))
        @test maxabsdif(center_of_gravity_with_covariance(
            :nonnegative, dat, win, c0...; maxiter=r.first, precision=wgt),
                        r.second) ≤ 1e-6
    end

end

# for i in 1:3; @printf("%d => (%.6f, %.6f, %.6f, %.6f, %.6f),\n", i, center_of_gravity_with_covariance(:nonnegative, dat, win, c0...; maxiter=i, precision=wgt)...); end
