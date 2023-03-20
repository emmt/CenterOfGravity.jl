"""

Module `SlidingWindows` provides slinding-windows useful to define
shift-invariant masks, weights, or convolutive filters.

"""
module SlidingWindows

export SlidingWindow, UniformlyTrue

"""
    UniformlyTrue()

is a singleton used as the weights of a simple Cartesian sliding window.

"""
struct UniformlyTrue end

"""
    SlidingWindow(arr)

yields a sliding window with weights given by the values of array `arr` and
with indices approximately centered along each dimension of `arr` (that is with
the same conventions as in `fftshift`).

A different centering can be chosen by specifying the ranges `rng...` of the
sliding window indices along all dimensions of `arr`:

    SlidingWindow(arr, rng...)

A Cartesian sliding window with all weights equal to `true` may be constructed
by:

    SlidingWindow(i...)

where `i...` are integers or integer-valued unit ranges.  If `i[k]` is an
integer, it is assumed to be the length of the `k`-th dimension of the sliding
window (indices will be approximately centered along that dimension); otherwise
`i[k]` specifies the range of indices along the `k`-th dimension of the sliding
window.

"""
struct SlidingWindow{T,N,P,R} <: AbstractArray{T,N}
    wgt::P             # array of weights
    inds::R            # array indices NTuple{N,AbstractUnitRange{Int}}
    off::NTuple{N,Int} # offsets along each dimensions
    dims::Dims{N}      # dimensions # FIXME: not needed?
end

# Accessors and abstract arrays API.
Base.parent(A::SlidingWindow) = getfield(A, :wgt)
Base.axes(A::SlidingWindow) = getfield(A, :inds)
offsets(A::SlidingWindow) = getfield(A, :off)
Base.size(A::SlidingWindow) = getfield(A, :dims)
Base.IndexStyle(::Type{<:SlidingWindow}) = IndexCartesian()

@inline function Base.getindex(A::SlidingWindow{T,N},
                               I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds getindex(parent(A), map(+, I, offsets(A))...)
end

@inline function Base.setindex!(A::SlidingWindow{T,N}, x,
                                I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(parent(A), x, map(+, I, offsets(A))...)
    return A
end

@inline function Base.getindex(A::SlidingWindow{Bool,N,UniformlyTrue},
                               I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(A, I...)
    return true
end

@inline function Base.setindex!(A::SlidingWindow{Bool,N,UniformlyTrue}, x,
                                I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(A, I...)
    error("attempt to set value of read-only sliding-window")
end


# FIXME: Similar aliases are in ArrayTools.
const SlidingWindowAxis = Union{Integer,AbstractUnitRange{<:Integer}}
const SlidingWindowAxes{N} = NTuple{N,SlidingWindowAxis}

# Constructors for uniformly true sliding windows.
SlidingWindow(inds::SlidingWindowAxis...) = SlidingWindow(inds)
SlidingWindow(inds::SlidingWindowAxes) = SlidingWindow(UniformlyTrue(), inds)
SlidingWindow(wgt::UniformlyTrue, inds::SlidingWindowAxes) =
    SlidingWindow(wgt, map(to_sliding_axis, inds))
function SlidingWindow(wgt::P,
                       inds::R) where {N,P<:UniformlyTrue,
                                       R<:NTuple{N,AbstractUnitRange{Int}}}
    dims = map(to_dimension, inds)
    off = ntuple(x -> 0, Val(N))
    return SlidingWindow{Bool,N,P,R}(wgt, inds, off, dims)
end

# Constructors for weighted sliding windows.
SlidingWindow(wgt::AbstractArray) =
    SlidingWindow(wgt, map(to_sliding_axis, size(wgt)))
SlidingWindow(wgt::AbstractArray{T,N}, inds::Vararg{SlidingWindowAxis,N}) where {T,N} =
    SlidingWindow(wgt, inds)
SlidingWindow(wgt::AbstractArray{T,N}, inds::SlidingWindowAxes{N}) where {T,N} =
    SlidingWindow(wgt, map(to_sliding_axis, inds))
function SlidingWindow(wgt::P,
                       inds::R) where {T,N,P<:AbstractArray{T,N},
                                       R<:NTuple{N,AbstractUnitRange{Int}}}
    dims = size(wgt)
    map(length, inds) == dims || error(
        "window indices must match the weights size")
    off = map((i,j) -> first(j) - first(i), inds, axes(wgt))
    return SlidingWindow{T,N,P,R}(wgt, inds, off, dims)
end

"""
    to_dimension(x)

converts argument `x` into an array dimension of type `Int`.

"""
to_dimension(x::Int) = x
to_dimension(x::Integer) = as(Int, x)
to_dimension(x::AbstractUnitRange{<:Integer}) = to_dimension(length(x))

"""
    to_sliding_axis(x)

converts argument `x` into a unit range of `Int` values and following the
conventions assumed by sliding windows.

"""
to_sliding_axis(x::AbstractUnitRange{Int}) = x
to_sliding_axis(x::AbstractUnitRange{<:Integer}) = as(Int, first(x)):as(Int, last(x))
to_sliding_axis(x::Base.OneTo{<:Integer}) = Base.OneTo{Int}(length(x))
to_sliding_axis(x::Integer) = begin
    # Use the same conventions as in `fftshift` to approximately center the
    # range of indices.
    n = max(as(Int, x), 0)
    i = -(n >> 1)
    return i:(n-1+i)
end

end # module
