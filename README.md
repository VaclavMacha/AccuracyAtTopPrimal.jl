# AccuracyAtTopPrimal.jl

To install this package use [Pkg REPL](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html) and the following command

```julia
 add https://github.com/VaclavMacha/AccuracyAtTopPrimal.jl
```

 ## Usage

This package provides a simple interface for solving many optimization problems with decision threshold constraints. The package provides two functions that can be used as an objectives for the optimization

- `fnr(targets, scores, t, [surrogate = quadratic])`: computes the approximation of false-negative rate
- `fpr(targets, scores, t, [surrogate = quadratic])`: computes the approximation of false-positive rate

where

- `targets` is a vector of targets (true labels),
- `scores` is a vector of classification scores given by the used model
- `t` is a decision threshold
- `surrogate` is a function that is used as an approximation of the indicator function (indicator function returns `1` if its argument is true and `0` otherwise).

The package also provides two basic surrogate functions

- `hinge(x, [ϑ = 1])`: hinge loss defined as ` max(0, 1 + ϑ * x)`
- `quadratic(x, [ϑ = 1])`: quadratic hinge loss defined as `max(0, 1 + ϑ * x)^2`

However, it is possible to define and use any other surrogate function. To define the decision threshold `t`, the package provides function `threshold(thres, targets, scores)` where the first argument specifies the model type. There are 8 different models supported by the package

- `PatMat`
- `PatMatNP`
- `TopPush`
- `TopPushK`
- `τFPL`
- `TopMean`
- `Grill`
- `GrillNP`

The following example shows, how to minimize false-negative rate with a given constraint that false-positive rate is smaller or equal to `5%`.

```julia
model = Chain(...) # Flux model
surrogate = hinge
thres = PatMatNP(0.05, surrogate)

function loss(data, targets)
    scores = model(data)
    t = threshold(thres, targets, scores)
    return fnr(target, scores, t, surrogate)
end
```
