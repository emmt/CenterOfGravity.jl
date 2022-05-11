# Portable random generator from "Numerical Recipes in C" by Press et al.
module PortableRandomGenerators

export SimpleRandomGenerator

using Random

const IA = 16807
const IM = 2147483647
const IQ = 127773
const IR = 2836
const MASK = 123459876

mutable struct SimpleRandomGenerator <: AbstractRNG
    val::Int32
    function SimpleRandomGenerator(seed::Integer)
        return new(xor(seed,MASK))
    end
end

function (rng::SimpleRandomGenerator)()
    val = rng.val
    # Compute val = (IA*val)%IM without overflows by Schrage’s method.
    k = div(val, IQ)
    val = IA*(val - k*IQ) - IR*k
    if val < 0
        val += IM
    end
    rng.val = val
    return val
end

(rng::SimpleRandomGenerator)(::Type{Int32}) = rng()
(rng::SimpleRandomGenerator)(T::Type{<:Integer}) = convert(T, rng())
(rng::SimpleRandomGenerator)(T::Type{<:AbstractFloat}) =
    convert(T, rng())/convert(T, IM)

function Random.rand(rng::SimpleRandomGenerator,
                     ::Random.SamplerType{T}) where {T}
    return rng(T)
end

function Random.randn(rng::SimpleRandomGenerator,
                      T::Type{<:AbstractFloat},
                      dims::Dims)
    return randn!(rng, Array{T}(undef, dims))
end

function Random.randn!(rng::SimpleRandomGenerator,
                       A::AbstractArray{T}) where {T<:AbstractFloat}
    v2 = zero(T)
    flag = false
    for i in eachindex(A)
        if flag
            A[i] = v2
            flag = false
        else
            v1, v2 = randn2(rng, T)
            A[i] = v1
            flag = true
        end
    end
    return A
end

# Box-Muller method for generating 2 normally distributed independent random
# variables.
function randn2(rng::SimpleRandomGenerator,
                T::Type{<:AbstractFloat})
    while true
        v1 = 2rng(T) - 1
        v2 = 2rng(T) - 1
        rsq = v1*v1 + v2*v2
        if 0 < rsq < 1
            scl = sqrt(-2log(rsq)/rsq)
            return (scl*v1, scl*v2)
        end
    end
end

end # module PortableRandomGenerators
