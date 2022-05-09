# CenterOfGravity [![Build Status](https://github.com/emmt/CenterOfGravity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/CenterOfGravity.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/CenterOfGravity.jl?svg=true)](https://ci.appveyor.com/project/emmt/CenterOfGravity-jl) [![Coverage](https://codecov.io/gh/emmt/CenterOfGravity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/CenterOfGravity.jl)

`CenterOfGravity` is a small Julia package to compute centers of gravity in
images (that is 2-dimensional arrays).  By playing with the arguments of the
`center_of_gravity` method, a variety of algorithms for computing the center of
gravity can be emulated by this package.


## Installation

The easiest way to install `CenterOfGravity` is via Julia registry
[`EmmtRegistry`](https://github.com/emmt/EmmtRegistry):

```julia
using Pkg
pkg"registry add General"  # if not yet any registries
pkg"registry add https://github.com/emmt/EmmtRegistry"
pkg"add CenterOfGravity"
```
