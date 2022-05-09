module CenterOfGravity

export
    SlidingWindow,
    WindowedArray,
    center_of_gravity

using CartesianBoxes

include("WindowedArrays.jl")
using .WindowedArrays

include("SlidingWindows.jl")
using .SlidingWindows
using .SlidingWindows: offsets

include("cog.jl")

end
