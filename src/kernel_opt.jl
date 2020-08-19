
export train_gpmodel


#####
# traditional log-likelihood
#####

function _loglikelihood(logw, sgp::SGP, indP::NTuple{2, Array{<:Array{<:Real,1},1}}) where SGP <: SparseGP    
    kernel = sgp.kernel
    X = indP[1]
    Y = indP[2]
    σ_n = sgp.σ_n
    return _loglikelihood(logw, kernel, X, Y, σ_n)
end

function _loglikelihood(logw, kernel, X, Y, σ_n)
    w = exp.(logw)
    vY = reshape(reduce(vcat, data.Y), :, length(X)*length(X[1]))
    
    ker = kernel(w)
    K = kernelmatrix(ker, X) + σ_n*I
    Kchol = cholesky(K)
    
    fitTerm = 1/2 * mapreduce(y -> y' * (Kchol \ y), +, eachrow(vY))
    detTerm = 2* sum(log.(diag(Kchol.L)))
    return fitTerm + detTerm
end


######
# FITC log likelihood cost
######
# add type to SparseGP and then (for NTuple{3,...}) dispatch on it for different likelihoods
function _loglikelihood(logw, sgp::SGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}} ) where SGP <: SparseGP
    kernel = sgp.kernel
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    σ_n = sgp.σ_n
    _loglikelihood(logw, kernel, Z, X, Y, σ_n)
end

function _loglikelihood(logw, kernel, Z, X, Y, σ_n)
    w = exp.(logw)
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    
    ker = kernel(w)
    Kff = kernelmatrix(ker, X)
    Kfu = kernelmatrix(ker, X, Z)
    Kuu = kernelmatrix(ker, Z)
    
    Qff = Symmetric(Kfu * ( Kuu \ Kfu' ))
    Λ = Diagonal(diag( Kff - Qff) .+ σ_n)
    QSChol = cholesky(Qff + Λ)
    
    fitTerm = 1/2 * mapreduce(y -> y' * (QSChol \ y), +, eachrow(vY))
    detTerm = 2* sum(log.(diag(QSChol.L)))
    return fitTerm + detTerm
end


#####
# gradient, for either of the two above
#####
function _llgrad(G, logw, sgp)
    tmp = gradient(w -> _loglikelihood(logw, sgp, sgp.inP), logw)
    G[:] = tmp[1]
end


######
# VLB, Titsias variatonal lower bound
######
function _variational_lowerbound(logw, sgp::SGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}}) where SGP <: SparseGP
    kernel = sgp.kernel
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    σ_n = sgp.σ_n
    _variational_lowerbound(logw, kernel, Z, X, Y, σ_n)
end

# add type to SparseGP and then (for NTuple{3,...}) dispatch on it for different likelihoods
function _variational_lowerbound(logw, kernel, Z, X, Y, σ_n)
    w = exp.(logw)
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    
    ker = kernel(w)
    Kff = kernelmatrix(ker, X)
    Kfu = kernelmatrix(ker, X, Z)
    Kuu = kernelmatrix(ker, Z)
    
    Qff = Symmetric(Kfu * ( Kuu \ Kfu' ))
    Λ =  σ_n*I
    QSChol = cholesky(Qff + Λ)
    T = Kff - Qff
    
    fitTerm = 1/2 * mapreduce(y -> y' * (QSChol \ y), +, eachrow(vY))
    detTerm = 2* sum(log.(diag(QSChol.L)))
    traceTerm = 1/(2*σ_n^2) * tr(T)
    return fitTerm + detTerm + traceTerm
end


function _vlbgrad(G, logw, sgp)
    tmp = gradient(w -> _loglikelihood(logw, sgp, sgp.inP), logw)
    G[:] = tmp[1]
end


abstract type SparseGPMethod end
struct FITC <: SparseGPMethod end
struct VLB <: SparseGPMethod end
# have this dispatch on LL/ elbo object

function define_objective(sgp::SGP; method::M = FITC(), grad = false) where {SGP <: SparseGP, M <: SparseGPMethod}
    return _define_objective(sgp, method, grad)
end

function _define_objective(sgp::SGP, method::FITC, grad = false) where {SGP <: SparseGP}
    c(logw) = _loglikelihood(logw, sgp, sgp.inP)
    if grad
        g(G,logw) = _llgrad(G, logw, sgp)
        return (c,g)
    else
        return c
    end
end


function _define_objective(sgp::SGP, method::VLB, grad = false) where {SGP <: SparseGP}
    c(logw) = _variational_lowerbound(logw, sgp, sgp.inP)
    if grad
        g(G,logw) = _vlbgrad(G, logw, sgp)
        return (c,g)
    else
        return c
    end
end



#####
# training function
#### 
function train_gpmodel(sgp::SGP; show_opt = false, method::M = FITC1(), grad = false) where {SGP <: SparseGP, M <: SparseGPMethod}
    ker = sgp.kernel
    obj = define_objective(sgp; method = method, grad = grad)
    optres = optimize(obj, log.(getparam(ker)))
    wopt = exp.(optres.minimizer)
    if show_opt
        display(optres)
        display(wopt)
    end
    optker = ker(wopt)
    
    return typeof(sgp)(optker, sgp.σ_n, sgp.inP, sgp.mean, sgp.trafo)
end