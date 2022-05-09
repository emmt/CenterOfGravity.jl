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

@testset "Center of gravity" begin
end
