# auxiliary functions
find_negatives(targets) = findall(vec(targets) .== 0)
find_positives(targets) = findall(vec(targets) .== 1)

@nograd find_negatives, find_positives

# surrogate functions
hinge(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)
quadratic(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)^2

# objectives
function fnr(targets, scores, t, surrogate = quadratic)
    return mean(surrogate.(t .- scores[find_positives(targets)]))
end

function fpr(targets, scores, t, surrogate = quadratic)
    return mean(surrogate.(scores[find_negatives(targets)] .- t))
end

# root finding
function find_root(f::Function, x_0::Real)
    try
        x = Roots.secant_method(f, x_0)
        isnan(x) && error("secant_method failed")
        return x
    catch
        try
            x = Roots.find_zero(f, (x_0, Inf))
            isnan(x) && error("find_zero failed")
            return x
        catch
            xrange = range(x_0; stop = 10*x_0, length = 100)
            xzero, ind  = findmin(abs.(f.(xrange)))
            return xrange[ind]
        end
    end
end
