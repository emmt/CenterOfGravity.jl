using CenterOfGravity
using Test, Random

@testset "Windowed arrays  " begin
    A = reshape(collect(1:20), (4,5))
    J = (2:3, 3:5)
    B = WindowedArray(A, J...)
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    @test axes(B) == J
    @test size(B) == map(length, J)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == [A[j1,j2] for j1 in J[1], j2 in J[2]]
end

@testset "Sliding windows  " begin
    # Sliding window from apprimately centered array.
    A = reshape(collect(1:20), (4,5))
    B = SlidingWindow(A)
    J = map(n -> -div(n,2) : n-1-div(n,2), size(A))
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    @test axes(B) == J
    @test size(B) == size(A)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == A

    # Sliding window with given axes.
    J = map(n -> Int16(0) : Int16(n-1), size(A))
    B = SlidingWindow(A, J...)
    @test eltype(B) === eltype(A)
    @test parent(B) === A
    @test axes(B) == J
    @test size(B) == size(A)
    @test [B[j1,j2] for j1 in J[1], j2 in J[2]] == A

    # Uniformly true mask.
    J = (-1:7, 0:4, 3:8)
    B = SlidingWindow(J...)
    @test eltype(B) === Bool
    @test parent(B) === CenterOfGravity.SlidingWindows.UniformlyTrue()
    @test axes(B) == J
    @test size(B) == map(length, J)
    @test [B[j1,j2,j3] for j1 in J[1], j2 in J[2], j3 in J[3]] == ones(Bool, size(B))
end

distance(a::NTuple{N,Real}, b::NTuple{N,Real}) where {N} =
    sqrt(sum(map((ca,cb) -> (ca - cb)^2, a, b)))

@testset "Center of gravity" begin
    # Model is a 2-D Gaussian.
    dims = (25,30)
    c = (9.01, 21.67) # central position
    w = 4.3 # full width at half max.
    σ = w/sqrt(8*log(2))
    mdl = [20*exp(-((x1 - c[1])^2 + (x2 - c[2])^2)/(2*σ^2))
           for x1 in 1:dims[1], x2 in 1:dims[2]]
    # Add some noise.
    rng = MersenneTwister(1234); # for reproducible results
    dat = mdl + 0.3.*randn(rng, Float64, size(mdl))
    # Compute center of gravity with uniformly true mask.
    c0 = map(n -> (1 + n)/2, dims) # initial position
    J = (-7:7, -7:7)
    win = SlidingWindow(J...)
    @test distance(center_of_gravity(dat, win, c0...; maxiter=1),
                   (9.0421, 21.0525)) ≤ 0.0003
    @test distance(center_of_gravity(dat, win, c0...; maxiter=2),
                   (8.9618, 21.6215)) ≤ 0.0003
    @test distance(center_of_gravity(dat, win, c0...; maxiter=3),
                   (8.9456, 21.6441)) ≤ 0.0003
    # Idem without negative pixels.
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=1),
                   (9.3411, 20.7550)) ≤ 0.0003
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=2),
                   (8.9775, 21.6089)) ≤ 0.0003
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=3),
                   (8.9653, 21.68145)) ≤ 0.0003
    # Compute center of gravity with uniform weights.
    win = SlidingWindow(ones(Float64, map(length, J)), J...)
    @test distance(center_of_gravity(dat, win, c0...; maxiter=1),
                   (9.0421, 21.0525)) ≤ 0.0003
    @test distance(center_of_gravity(dat, win, c0...; maxiter=2),
                   (8.9618, 21.6215)) ≤ 0.0003
    @test distance(center_of_gravity(dat, win, c0...; maxiter=3),
                   (8.9456, 21.6441)) ≤ 0.0003
    # Idem without negative pixels.
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=1),
                   (9.3411, 20.7550)) ≤ 0.0003
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=2),
                   (8.9775, 21.6089)) ≤ 0.0003
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=3),
                   (8.9653, 21.68145)) ≤ 0.0003
    # Compute center of gravity with uniform weights without corners.
    msk = [round(Int, sqrt(x1^2 + x2^2)) ≤ 7 for x1 in J[1], x2 in J[2]]
    win = SlidingWindow(msk)
    @test distance(center_of_gravity(dat, win, c0...; maxiter=1),
                   (9.6735, 20.6513)) ≤ 0.0003
    @test distance(center_of_gravity(dat, win, c0...; maxiter=2),
                   (8.9344, 21.6389)) ≤ 0.0003
    @test distance(center_of_gravity(dat, win, c0...; maxiter=3),
                   (8.9803, 21.6544)) ≤ 0.0003
    # Idem without negative pixels.
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=1),
                   (9.9468, 20.3066)) ≤ 0.0003
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=2),
                   (9.0167, 21.5738)) ≤ 0.0003
    @test distance(center_of_gravity(:nonnegative, dat, win, c0...; maxiter=3),
                   (8.9821, 21.6671)) ≤ 0.0003
end
