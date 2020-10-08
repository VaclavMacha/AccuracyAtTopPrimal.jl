module AccuracyAtTopPrimal

using LinearAlgebra, Statistics, Flux, Roots, Parameters

using Zygote: @adjoint, @nograd

export PatMat, PatMatNP, TopPush, TopPushK, τFPL, TopMean, Grill, GrillNP
export fnr, fpr, hinge, quadratic, threshold

abstract type AbstractThreshold end

include("thresholds.jl")
include("utilities.jl")

end # module
