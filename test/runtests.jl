using CenterOfGravity
using Test

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

@testset "Center of gravity" begin
end
