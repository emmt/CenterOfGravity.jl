# Allowed types for specifying a 2-D position as a single argument.
const Position = Union{CartesianIndex{2},Tuple{Real,Real}}

"""
    to_position(T, pos)

yields position `pos` converted to 2-tuple of coordinates of floating-point
type `T` as required by the `center_of_gravity` method.

"""
to_position(::Type{T}, pos::NTuple{2,T}) where {T} = pos
to_position(::Type{T}, pos::NTuple{2,Real}) where {T} =
    (to_type(T, pos[1]), to_type(T, pos[2]))
to_position(::Type{T}, pos::CartesianIndex{2}) where {T} =
    to_position(T, Tuple(pos))

"""
    to_algorithm(alg)

yields algorithm `alg` converted to the king of instance required by the
`center_of_gravity` method.

"""
to_algorithm(alg::Val) = alg
to_algorithm(alg::Symbol) = Val(alg)

"""
    to_type(T, x) -> x′::T

converts argument `x` to type `T`.  Compared to `convert(T,x)` a type assertion
is imposed.

"""
to_type(::Type{T}, x::T) where {T} = x
to_type(::Type{T}, x) where {T} = convert(T, x)::T

"""
    center_of_gravity([alg = :simple,] img, win, pos) -> cog

yields the center of gravity in image `img` using a sliding window `win`
centered at initial position `pos` (rounded to nearest pixel) to select and
weight the pixels taken into account for computing the center of gravity.

The initial and final positions `pos` and `cog` are given in fractional index
units relative to the image array `img`.  The initial position `pos` may be
specified in a variety of forms (2 reals, a 2-tuple of reals, a 2-dimensional
Cartesian index, etc.), but the result `cog` is always a 2-tuple of
floating-point values.

The keyword `maxiter` (1 by default) may be used to specify the maximum number
of iterations of the algorithm (at least one iteration is performed).  At each
iteration, the algorithm updates the position of the center of gravity by
moving the sliding window at the previous estimate of this position (initially
specified by argument `pos`).  The `k`-th coordinate of the new center of
gravity is computed as:

    cog[k] = (sum_{i ∈ R} img[i]*win[i-j]*x[i][k])/
        (sum_{i ∈ R} img[i]*win[i-j])

where `R` gives the indices of the pixels take into account, `j` is the offset
to best center the sliding window at the previously estimated position of the
center of gravity, and `x[i][j]` denotes the `k`-th component of the position
at index `i` in the image `img`.  The set `R` depends on the offset `j` and on
the values of the image and of the sliding window in the intersection of their
supports when the sliding window is moved by offset `j` (see below).

Note that, if the sliding window is a simple mask (its elements are booleans),
the above formula amounts to computing the usual center of gravity.  If the
sliding window has values, the *weighted* center of gravity is computed.

Optional argument `alg` specifies the algorithm to use, it has the form `sym`
or `Val(sym)` where `sym` is one of the following symbols:

- `:simple` to compute the center of gravity on all values inside the sliding
  window.  This is equivalent to having the set `R` containing indices `i` such
  that `img[i]` and `win[i-j]` are in bounds.

- `:nonnegative` to compute the center of gravity on all nonnegative values
  inside the sliding window.  This is equivalent to having the set `R`
  containing indices `i` such that `img[i]` and `win[i-j]` are in bound,
  and such that `img[i] > 0` and `win[i-j] > 0`.

Use a `WindowedArray` instead of a `view` for the input image if you want to
restrict the computations to a rectangular region of the image while preserving
image coordinates:

    center_of_gravity(WindowedArray(img, 21:63, 23:65), win, pos)

For consistency with [`center_of_gravity_with_covariance`](@ref), keyword
`precision` can be specified to provide the pixel-wise precision of the image
`img`.  If specified, all its entries must be nonnegative (this is not checked)
and only pixels with non-zero precision are taken into account in the
computation of the center of gravity (in addition of other selection rules).

!!! note
    If algorithm is fully specified, i.e. as `alg = Val(:...)`, the code does
    not allocate any memory.  This is of importance for real-time applications.

""" center_of_gravity

# Initial position specified as 2 coordinates.
function center_of_gravity(img::AbstractMatrix{T},
                           win::SlidingWindow{<:Union{T,Bool},2},
                           pos::Vararg{Real,2};
                           kwds...) where {T<:AbstractFloat}
    return center_of_gravity(img, win, pos; kwds...)
end
function center_of_gravity(alg::Union{Symbol,Val},
                           img::AbstractMatrix{T},
                           win::SlidingWindow{<:Union{T,Bool},2},
                           pos::Vararg{Real,2};
                           kwds...) where {T<:AbstractFloat}
    return center_of_gravity(alg, img, win, pos; kwds...)
end

# Provide default algorithm.
function center_of_gravity(img::AbstractMatrix{T},
                           win::SlidingWindow{<:Union{T,Bool},2},
                           pos::Position;
                           kwds...) where {T<:AbstractFloat}
    return center_of_gravity(:simple, img, win, pos; kwds...)
end

# Convert algorithm and initial position.
function center_of_gravity(alg::Union{Symbol,Val},
                           img::AbstractMatrix{T},
                           win::SlidingWindow{<:Union{T,Bool},2},
                           pos::Position;
                           kwds...) where {T<:AbstractFloat}
    return center_of_gravity(to_algorithm(alg), img, win,
                             to_position(T, pos); kwds...)
end

# NOTE: This version is able to deliver ~ 10.7 Gflops on an AMD Ryzen
#       Threadripper 2950X CPU (with 16 cores but a single thread is used) with
#       the most simple algorithm (assuming 10 operations/pixel).  This does
#       not depend much on the algorithm (assuming 12 operations/pixel for the
#       most complex one) nor on the size of the sliding window (15×15 and
#       11×11 were tried).  The speed is the same in single and double
#       precision.  Forcing SIMD worsen the performances so we do not use
#       `@simd`.  A more basic version not using closures nor lambda functions
#       was found to be as fast, so we keep the version with lambda functions
#       as it is much more flexible and yet readable (only the computation of
#       the "weights" depends on the algorithm).  The same kind of conclusions
#       have been made for the `LocalFilter` package.
function center_of_gravity(
    alg::Val,
    img::AbstractMatrix{T},
    win::SlidingWindow{<:Union{T,Bool},2},
    pos::NTuple{2,T};
    precision::Union{Nothing,AbstractMatrix{T}} = nothing,
    maxiter::Int = 1,
    verbose::Bool = false) where {T<:AbstractFloat}

    # Extract the "supports" of the image and sliding window.
    img_box = CartesianBox(img) # image support
    win_box = CartesianBox(win) # sliding window support
    if precision !== nothing
        CartesianBox(precision) == img_box || error(
            "array of pixel precisions and image have different indices")
    end

    # Split initial position in coordinates and round them to the nearest
    # pixel.
    c1, c2 = pos
    j1, j2 = round(Int, c1), round(Int, c2)

    # Iterate algorithm.
    z = zero(T)
    iter = 0
    verbose && println("iter: $iter, cog: ($c1, $c2)")
    while true
        # Determine the region `R` such that `∀ i ∈ R`, `img[i]` and `win[i-j]`
        # are both valid with `j = (j1,j2)` the position of the center of the
        # sliding window relative to the image.
        R = img_box ∩ (win_box + (j1,j2)) # ROI for indices `i`

        # Compute the terms needed by the center of gravity.  To avoid adding
        # multiple index offsets when indexing the sliding window, we note
        # that:
        #
        #    win[i - j] -> parent(win)[i - (j - k)]
        #
        # with `k = offsets(win)`.
        I1, I2 = Tuple(R) # split ROI along dimensions
        k1, k2 = offsets(win) # split offsets along dimensions
        u0, u1, u2 = cog_loop(
            precision, img, parent(win),
            I1, j1 - k1, c1,
            I2, j2 - k2, c2,
            (z, z, z), # initial state
            (state, p, a, b, x1, x2) -> (
                w = cog_weight(alg, p, a, b);
                return (state[1] + w,
                        state[2] + w*x1,
                        state[3] + w*x2)))

        # Update center of gravity.
        if u0 ≤ z
            # No changes.
            break
        end
        c1 += u1/u0
        c2 += u2/u0

        # Increment number of iterations.
        iter += 1
        verbose && println("iter: $iter, cog: ($c1, $c2)")
        if iter ≥ maxiter
            # Maximum number of iterations have been reached.
            break
        end

        # Update position of the sliding window.
        j1_prev, j1 = j1, round(Int, c1)
        j2_prev, j2 = j2, round(Int, c2)
        if (j1 == j1_prev)&(j2 == j2_prev)
            # Search has converged because the region of interest will remain
            # the same.
            break
        end
    end
    return (c1, c2)
end

"""
    center_of_gravity_with_covariance([alg = :simple,] img, win, pos)
    -> (x1, x2, c11, c12, c22)

yields the center of gravity and its covariance in image `img` using a sliding
window `win` centered at initial position `pos` (rounded to nearest pixel) to
select and weight the pixels taken into account for computing the center of
gravity.

The result is the 5-tuple `(x1,x2,c11,c12,c22)` where `(x1,x2)` are the
coordinates of the center of gravity while `[c11 c12; c12 c22]` is the
associated covariance matrix.

See [`center_of_gravity`](@ref) for a description of common parameters.

Keyword `precision` can be specified to provide the pixel-wise precision of the
image `img`.  If specified, all its entries must be nonnegative (this is not
checked) and only pixels with non-zero precision are taken into account in the
computation of the center of gravity (in addition of other selection rules).
If not specified, all pixels are assumed valid, the data noise is assumed
i.i.d. (independent and identically distributed), and the returned covariance
must be scaled by the noise variance.

""" center_of_gravity_with_covariance

# Initial position specified as 2 coordinates.
function center_of_gravity_with_covariance(
    img::AbstractMatrix{T},
    win::SlidingWindow{<:Union{T,Bool},2},
    pos::Vararg{Real,2};
    kwds...) where {T<:AbstractFloat}
    return center_of_gravity_with_covariance(img, win, pos; kwds...)
end
function center_of_gravity_with_covariance(
    alg::Union{Symbol,Val},
    img::AbstractMatrix{T},
    win::SlidingWindow{<:Union{T,Bool},2},
    pos::Vararg{Real,2};
    kwds...) where {T<:AbstractFloat}
    return center_of_gravity_with_covariance(alg, img, win, pos; kwds...)
end

# Provide default algorithm.
function center_of_gravity_with_covariance(
    img::AbstractMatrix{T},
    win::SlidingWindow{<:Union{T,Bool},2},
    pos::Position;
    kwds...) where {T<:AbstractFloat}
    return center_of_gravity_with_covariance(:simple, img, win, pos; kwds...)
end

# Convert algorithm and initial position.
function center_of_gravity_with_covariance(
    alg::Union{Symbol,Val},
    img::AbstractMatrix{T},
    win::SlidingWindow{<:Union{T,Bool},2},
    pos::Position;
    kwds...) where {T<:AbstractFloat}
    return center_of_gravity_with_covariance(to_algorithm(alg), img, win,
                                             to_position(T, pos); kwds...)
end

function center_of_gravity_with_covariance(
    alg::Val,
    img::AbstractMatrix{T},
    win::SlidingWindow{<:Union{T,Bool},2},
    pos::NTuple{2,T};
    precision::Union{Nothing,AbstractMatrix{T}} = nothing,
    maxiter::Int = 1,
    verbose::Bool = false) where {T<:AbstractFloat}

    # Extract the "supports" of the image and sliding window.
    img_box = CartesianBox(img) # image support
    win_box = CartesianBox(win) # sliding window support
    if precision !== nothing
        CartesianBox(precision) == img_box || error(
            "array of pixel precisions and image have different indices")
    end

    # Split initial position in coordinates and round them to the nearest
    # pixel.
    c1, c2 = pos
    j1, j2 = round(Int, c1), round(Int, c2)

    # Iterate algorithm.
    z = zero(T)
    iter = 0
    verbose && println("iter: $iter, cog: ($c1, $c2)")
    while true
        # Determine the region `R` such that `∀ i ∈ R`, `img[i]` and `win[i-j]`
        # are both valid with `j = (j1,j2)` the position of the center of the
        # sliding window relative to the image.
        R = img_box ∩ (win_box + (j1,j2)) # ROI for indices `i`

        # Compute the terms needed by the center of gravity.  To avoid adding
        # multiple index offsets when indexing the sliding window, we note
        # that:
        #
        #    win[i - j] -> parent(win)[i - (j - k)]
        #
        # with `k = offsets(win)`.
        I1, I2 = Tuple(R) # split ROI along dimensions
        k1, k2 = offsets(win) # split offsets along dimensions
        u0, u1, u2 = cog_loop(
            precision, img, parent(win),
            I1, j1 - k1, c1,
            I2, j2 - k2, c2,
            (z, z, z), # initial state
            (state, p, a, b, x1, x2) -> (
                w = cog_weight(alg, p, a, b);
                return (state[1] + w,
                        state[2] + w*x1,
                        state[3] + w*x2)))

        # Update center of gravity.
        if u0 ≤ z
            # No changes.
            return (c1, c2, T(Inf), z, T(Inf))
        end
        c1 += u1/u0
        c2 += u2/u0

        # Update position of the sliding window and assume convergence if the
        # sliding window will not move.
        j1_prev, j1 = j1, round(Int, c1)
        j2_prev, j2 = j2, round(Int, c2)
        final = (j1 == j1_prev)&(j2 == j2_prev)

        # Increment number of iterations and assume convergence if the maximum
        # number of iterations have been reached.
        iter += 1
        verbose && println("iter: $iter, cog: ($c1, $c2)")
        final |= (iter ≥ maxiter)

        if final
            # Compute quantities needed for the covariance.
            C = cog_loop(
                precision, img, parent(win),
                I1, j1 - k1, c1,
                I2, j2 - k2, c2,
                (z, z, z, z, z, z),
                (C, p, a, b, x1, x2) -> (
                    w = cog_cov_weight(alg, p, a, b);
                    # FIXME: factorize?
                    return (C[1] + w,
                            C[2] + w*x1,
                            C[3] + w*x2,
                            C[4] + w*x1*x1,
                            C[5] + w*x1*x2,
                            C[6] + w*x2*x2)))
            # Extract the coefficients of the covariance matrix Cu of
            # u = (u0,u1,u2)' which is:
            #
            #     Cu = [C[1]  C[2]  C[3];
            #           C[2]  C[4]  C[5];
            #           C[3]  C[5]  C[6]]
            #
            C1, C2, C3, C4, C5, C6 = C
            #
            # Compute the coefficients J1, J2, and J3 of the Jacobian matrix
            # ∂x/∂u with x = (c1,c2)' which is:
            #
            #     J = [J1 J2 0; J3 0 J2] = [-u1/u0^2  1/u0  0;
            #                               -u2/u0^2   0   1/u0]
            J1 = -u1/u0^2
            J2 = 1/u0
            J3 = -u2/u0^2
            #
            # Return the center of gravity and the coefficients of
            # the covariance matrix Cx = J*Cu*J' of x.
            #
            return (c1, c2,
                    C4*J2^2 + 2*C2*J1*J2 + C1*J1^2,
                    (C2*J2 + C1*J1)*J3 + C5*J2^2 + C3*J1*J2,
                    C1*J3^2 + 2*C3*J2*J3 + C6*J2^2)
        end
    end
end

"""
    center_of_gravity_with_precision([alg = :simple,] img, win, pos)
    -> (x1, x2, w11, w12, w22)

yields the center of gravity and its precision matrix in image `img` using a
sliding window `win` centered at initial position `pos` (rounded to nearest
pixel) to select and weight the pixels taken into account for computing the
center of gravity.

The result is the 5-tuple `(x1,x2,w11,w12,w22)` where `(x1,x2)` are the
coordinates of the center of gravity while `[w11 w12; w12 w22]` is the
associated precision matrix.

See [`center_of_gravity_with_covariance`](@ref) for a description of common
parameters and their interpretation to compute the center of gravity and
associated covariance matrix.

"""
function center_of_gravity_with_precision(args...; kwds...)
    x1, x2, C_11, C_12, C_22 = center_of_gravity_with_covariance(
        args...; kwds...)
    q = 1/(C_11*C_22 - C_12^2)
    return (x1, x2, q*C_22, -q*C_12, q*C_11)
end

"""
    cog_weight(alg, [p,] a, b) -> w

yields the weight of a given pixel in the terms required by the centrer of
gravity as computed by algorithm `alg` and with `p ≥ 0` the precision of the
pixel (optional), `a` the pixel intensity and `b` the corresponding value in
the sliding window.  Typically `w = a*b`.  If the precision `p` is `nothing`,
it is assumed that all pixels have non-zero precision.

"""
cog_weight(::Val{:simple}, ::Nothing, a::T, b::T) where {T<:AbstractFloat} = a*b

# NOTE: We use `ifelse` and `&` (not `&&`) to avoid branching.
cog_weight(::Val{:simple}, ::Nothing, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse(b, a, zero(T))

cog_weight(::Val{:nonnegative}, ::Nothing, a::T, b::T) where {T<:AbstractFloat} =
    ifelse((a > zero(T))&(b > zero(T)), a*b, zero(T))

cog_weight(::Val{:nonnegative}, ::Nothing, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse((a > zero(T))&b, a, zero(T))

# Idem but with precision specified (p ≥ 0 is assumed).
cog_weight(::Val{:simple}, p::T, a::T, b::T) where {T<:AbstractFloat} =
    ifelse((p > zero(T)), a*b, zero(T))

cog_weight(::Val{:simple}, p::T, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse((p > zero(T))&b, a, zero(T))

cog_weight(::Val{:nonnegative}, p::T, a::T, b::T) where {T<:AbstractFloat} =
    ifelse((p > zero(T))&(a > zero(T))&(b > zero(T)), a*b, zero(T))

cog_weight(::Val{:nonnegative}, p::T, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse((p > zero(T))&(a > zero(T))&b, a, zero(T))

"""
    cog_cov_weight(alg, p, a, b) -> w

yields the weight of a given pixel in the terms required to estimate the
covariance of the centrer of gravity as computed by algorithm `alg` and with `p
≥ 0` the precision of the pixel (optional), `a` the pixel intensity, and `b`
the corresponding value in the sliding window.  Typically `w = b^2/p`.  If the
precision `p` is `nothing`, it is assumed that the noise is i.i.d. and the
resulting covariance has to be multiplied by the variance of the noise.

"""
cog_cov_weight(::Val{:simple}, ::Nothing, a::T, b::T) where {T<:AbstractFloat} = b^2

cog_cov_weight(::Val{:simple}, ::Nothing, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse(b, one(T), zero(T))

cog_cov_weight(::Val{:nonnegative}, ::Nothing, a::T, b::T) where {T<:AbstractFloat} =
    ifelse((a > zero(T))&(b > zero(T)), b^2, zero(T))

cog_cov_weight(::Val{:nonnegative}, ::Nothing, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse((a > zero(T))&b, one(T), zero(T))

# Idem but with precision specified (p ≥ 0 is assumed).
cog_cov_weight(::Val{:simple}, p::T, a::T, b::T) where {T<:AbstractFloat} =
    ifelse((p > zero(T)), b^2/p, zero(T))

cog_cov_weight(::Val{:simple}, p::T, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse((p > zero(T))&b, one(T)/p, zero(T))

cog_cov_weight(::Val{:nonnegative}, p::T, a::T, b::T) where {T<:AbstractFloat} =
    ifelse((p > zero(T))&(a > zero(T))&(b > zero(T)), b^2/p, zero(T))

cog_cov_weight(::Val{:nonnegative}, p::T, a::T, b::Bool) where {T<:AbstractFloat} =
    ifelse((p > zero(T))&(a > zero(T))&b, one(T)/p, zero(T))

"""
    cog_loop([P,] A, B, I1,j1,c1, I2,j2,c2, state, update)

runs the loop for computing the center of gravity (COG).  This is equivalent to:

    for i2 in I2, i1 in I1
        state = update(state, A[i1,i2], B[i1-j1,i2-j2], i1 - c1, i2 - c2)
    end
    return state

If argument `P` is specifed then:

    for i2 in I2, i1 in I1
        state = update(state, P[i1,i2], A[i1,i2], B[i1-j1,i2-j2], i1 - c1, i2 - c2)
    end
    return state

"""
function cog_loop(::Nothing,
                  A::AbstractMatrix{T},
                  B::UniformlyTrue,
                  I1::AbstractUnitRange{Int}, j1::Int, c1::T,
                  I2::AbstractUnitRange{Int}, j2::Int, c2::T,
                  state, update) where {T<:AbstractFloat}
    @inbounds for i2 in I2
        x2 = T(i2) - c2
        for i1 in I1 # FIXME: slower with `@simd`
            x1 = T(i1) - c1
            state = update(state, nothing, A[i1,i2], true, x1, x2)
        end
    end
    return state
end

function cog_loop(::Nothing,
                  A::AbstractMatrix{T},
                  B::AbstractMatrix,
                  I1::AbstractUnitRange{Int}, j1::Int, c1::T,
                  I2::AbstractUnitRange{Int}, j2::Int, c2::T,
                  state, update) where {T<:AbstractFloat}
    @inbounds for i2 in I2
        x2 = T(i2) - c2
        for i1 in I1 # FIXME: slower with `@simd`
            x1 = T(i1) - c1
            state = update(state, nothing, A[i1,i2], B[i1-j1,i2-j2], x1, x2)
        end
    end
    return state
end

function cog_loop(P::AbstractMatrix{T},
                  A::AbstractMatrix{T},
                  B::UniformlyTrue,
                  I1::AbstractUnitRange{Int}, j1::Int, c1::T,
                  I2::AbstractUnitRange{Int}, j2::Int, c2::T,
                  state, update) where {T<:AbstractFloat}
    @inbounds for i2 in I2
        x2 = T(i2) - c2
        for i1 in I1 # FIXME: slower with `@simd`
            x1 = T(i1) - c1
            state = update(state, P[i1,i2], A[i1,i2], true, x1, x2)
        end
    end
    return state
end

function cog_loop(P::AbstractMatrix{T},
                  A::AbstractMatrix{T},
                  B::AbstractMatrix,
                  I1::AbstractUnitRange{Int}, j1::Int, c1::T,
                  I2::AbstractUnitRange{Int}, j2::Int, c2::T,
                  state, update) where {T<:AbstractFloat}
    @inbounds for i2 in I2
        x2 = T(i2) - c2
        for i1 in I1 # FIXME: slower with `@simd`
            x1 = T(i1) - c1
            state = update(state, P[i1,i2], A[i1,i2], B[i1-j1,i2-j2], x1, x2)
        end
    end
    return state
end
