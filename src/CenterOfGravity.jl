module CenterOfGravity

export
    SlidingWindow,
    WindowedArray,
    center_of_gravity,
    center_of_gravity_with_covariance,
    center_of_gravity_with_precision

using TypeUtils, CartesianBoxes

include("WindowedArrays.jl")
using .WindowedArrays

include("SlidingWindows.jl")
using .SlidingWindows
using .SlidingWindows: offsets

include("cog.jl")

end
