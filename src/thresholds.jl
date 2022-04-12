function threshold(thres::AbstractThreshold, targets, scores)
    return find_threshold(thres, cpu(targets), cpu(scores))[1]
end

@adjoint function threshold(thres::AbstractThreshold, targets, scores)
    t, Δt_s = find_threshold(thres, cpu(targets), cpu(scores))
    t = convert(eltype(scores), t)
    Δt_s = convert(typeof(scores), Δt_s)
    return t, Δ -> (nothing, nothing, Δ .* Δt_s)
end

# methods
quant(surrogate, x, τ) = sum(surrogate, x)/length(x) - τ

struct PatMat <: AbstractThreshold
    τ::Real
    surrogate::Function
end

function find_threshold(thres::PatMat, targets, scores)
    τ = thres.τ
    surrogate = thres.surrogate

    t = [find_root(t -> quant(surrogate, scores .- t, τ), quantile(vec(scores), 1 - τ))]
    ps = params(scores, t)
    ∇l = gradient(() -> quant(surrogate, scores .- t, τ), ps)

    return t[1], .-∇l[scores] ./ ∇l[t]
end

struct PatMatNP <: AbstractThreshold
    τ::Real
    surrogate::Function
end

function find_threshold(thres::PatMatNP, targets, scores)
    τ = thres.τ
    surrogate = thres.surrogate
    inds = find_negatives(targets)
    s = @views scores[inds]

    t = [find_root(t -> quant(surrogate, s .- t, τ), quantile(vec(s), 1 - τ))]
    ps = params(s, t)
    ∇l = gradient(() -> quant(surrogate, s .- t, τ), ps)

    ∇l_s = zero(scores)
    ∇l_s[inds] .= ∇l[s]

    return t[1], .-∇l_s ./ ∇l[t]
end

struct TopPush <: AbstractThreshold end

function find_threshold(::TopPush, targets, scores)
    inds = find_negatives(targets)
    s = @views scores[inds]

    t, ind = findmax(s)
    Δt_s = zero(scores)
    Δt_s[inds[ind]] = 1

    return t, Δt_s
end

struct TopPushK <: AbstractThreshold
    K::Int
end

function find_threshold(thres::TopPushK, targets, scores)
    K = thres.K
    inds = find_negatives(targets)
    s = @views scores[inds]

    ind = partialsortperm(s, 1:K; rev=true)
    Δt_s = zero(scores)
    Δt_s[inds[ind]] .= 1 / K

    return sum(s[ind]) / K, Δt_s
end

struct τFPL <: AbstractThreshold
    τ::Real
end

function find_threshold(thres::τFPL, targets, scores)
    τ = thres.τ
    inds = find_negatives(targets)
    s = scores[inds]
    K = floor(Int64, τ*length(inds)) + 1

    ind = partialsortperm(s, 1:K; rev = true)
    Δt_s = zero(scores)
    Δt_s[inds[ind]] .= 1/K

    return sum(s[ind])/K, Δt_s
end

struct TopMean <: AbstractThreshold
    τ::Real
end

function find_threshold(thres::TopMean, targets, scores)
    τ = thres.τ
    K = floor(Int64, τ*length(scores)) + 1

    ind = partialsortperm(vec(scores), 1:K; rev = true)
    Δt_s = zero(scores)
    Δt_s[ind] .= 1/K

    return sum(scores[ind])/K, Δt_s
end

struct Grill <: AbstractThreshold
    τ::Real
end

function find_threshold(thres::Grill, targets, scores)
    τ = thres.τ
    t = quantile(vec(scores), 1 - τ)
    Δt_s = zero(scores)

    return t, Δt_s
end

struct GrillNP <: AbstractThreshold
    τ::Real
end

function find_threshold(thres::GrillNP, targets, scores)
    τ = thres.τ
    inds = find_negatives(targets)
    t = quantile(scores[inds], 1 - τ)
    Δt_s = zero(scores)

    return t, Δt_s
end
