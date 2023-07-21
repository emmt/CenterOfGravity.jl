"""

Module `WindowedArrays` provides array-like objects corresponding to Cartesian
sub-regions of some other arrays and preserving their indices.

"""
module WindowedArrays

export WindowedArray

using TypeUtils

"""
    B = WindowedArray(A, J1, ..., JN)

yields an array-like object `B` corresponding to `A[J1,...,JN]` where `A` is an
`N`-dimensional array while `J1`, ..., `JN` are colons or integer-valued
unit-ranges which specify the ranges to take along the dimensions of `A`. The
returned object shares its contents with `A`.

The sub-region may also be specified as an `N`-tuple `(J1,...,JN)`.

The difference with the result of `view(A,J1,...,JN)` is that the indices in
the windowed array are the same (but restricted to a smaller region) as in the
orginal array. The same kind of object could also be built by composing a
*view* and an *offset array* provided by the `OffsetArrays` package.

Call `parent(B)` to get the array storing the values of a windowed array `B`.

Call `axes(B)` to get the index ranges of a windowed array `B`.

"""
struct WindowedArray{T,N,P<:AbstractArray{T,N},
                     I<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractArray{T,N}
    # Type parameters:
    #   T = element type
    #   N = number of dimensions
    #   P = type of parent object backing the storage
    #   I = type of indices
    parent::P
    indices::I
    # "Private" inner constructor to prevent building objects with unchecked
    # contents.
    function WindowedArray(::Val{:inbounds}, parent::P,
                           indices::I) where {T,N,P<:AbstractArray{T,N},
                                              I<:NTuple{N,AbstractUnitRange{Int}}}
        return new{T,N,P,I}(parent, indices)
    end
end

# Accessors and array API (only Cartesian indexing is supported for windowed
# arrays).
Base.parent(A::WindowedArray) = getfield(A, :parent)
Base.axes(A::WindowedArray) = getfield(A, :indices)
Base.size(A::WindowedArray) = map(length, axes(A))
Base.IndexStyle(::Type{<:WindowedArray}) = IndexCartesian()

@inline function Base.getindex(A::WindowedArray{T,N},
                               I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds getindex(parent(A), I...)
end

@inline function Base.setindex!(A::WindowedArray{T,N}, x,
                                I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(parent(A), x, I...)
    return A
end

# Union of types allowed for specifying windowed array index ranges.
const AxisRange = Union{Colon,AbstractUnitRange{<:Integer}}

# Constructors.
#
# We purposely do no provide constructors with specified type parameters like:
#
#     WindowedArray{T}(A, J1, ..., JN)
#
# because the returned object would not share its contents with `A` if
# conversion occurs.

WindowedArray(A::AbstractArray{T,N}, J::Vararg{AxisRange,N}) where {T,N} =
    WindowedArray(A, J)

function WindowedArray(A::AbstractArray{T,N},
                       J::NTuple{N,AxisRange}) where {T,N}
    # Private method to check and convert sub-ranges for windowed array.
    subrange(I::AbstractUnitRange{Int}, ::Colon) = I
    subrange(I::AbstractUnitRange{<:Integer}, ::Colon) =
        as(Int,first(I)):as(Int,last(I))
    subrange(I::Base.OneTo{<:Integer}, ::Colon) = Base.OneTo{Int}(length(I))
    function subrange(I::AbstractUnitRange{<:Integer},
                      J::AbstractUnitRange{<:Integer})
        J ⊆ I || error("out of range sub-range for windowed array")
        return subrange(J, :)
    end

    return WindowedArray(Val(:inbounds), A, map(subrange, axes(A), J))
end

end # module
