module AccuracyAtTopPrimal

using Flux
using LinearAlgebra
using Roots
using Statistics

using Flux: params
using Zygote: @adjoint, @nograd

export PatMat
export PatMatNP
export TopPush
export TopPushK
export τFPL
export TopMean
export Grill
export GrillNP
export fnr
export fpr
export hinge
export quadratic
export threshold

abstract type AbstractThreshold end

include("thresholds.jl")
include("utilities.jl")

end # module
