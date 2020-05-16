export globalcost, localcost
export cost, diff, detailloglikelihood
export computeinitU, initcost, optinitU

function optinitU(sigma0, y, ker)
    initU = computeinitU(y, ker)
    initopt = Optim.optimize(sig -> -npODEs.initcost(sig, y, ker), sigma0)

    println("Found $(initopt.minimizer) after $(initopt.iterations) iterations")
    initoptsigma = initopt.minimizer
end


function computeinitU(y, ker)
    tZ = y.u[1:end-1];
    tker = npODEs.kernel(tZ, ker.kerneltype);

    # TODO: higher order difference quotient
    tU = diff(y.u) ./ diff(y.t);
    tnpODE = npODE(tU, tker);

    initU = evalgpode.(ker.Z, Ref(tnpODE))
end

function initcost(sigma, y, ker)
    if any(sigma .< 0.01)
        return -1000
    end

    initker = npODEs.kernel(ker.Z, typeof(ker.kerneltype)(sigma))
    initU = computeinitU(y, initker)
    
    npODEs.loglpu(initU, initker)
end

function loglpu(U, kernel)
    vU = reduce(vcat, U)
    return -1/2 * vU' * (kernel.Kchol \ vU) - 1/2 * log(det(kernel.Kchol)) 
end

abstract type costtype end
struct globalcost <: costtype end
struct localcost <: costtype end

function trajdiff(w, y, npode, ::localcost)
    N = length(y.t)-1 # should be optimized

    diffts = diff(y.t);
    initValues = y.u[1:end-1]
    # initial values don't matter, will be overwritten anyway
    x0 = ones(length(y.u[1]))
    tspan = 1.0
    gpprob = ODEProblem(gpode!,x0, tspan, npode)
    ensemble = EnsembleProblem(gpprob; 
        output_func = (sol,i) -> (sol.u[end], false),
        prob_func = (prob, i, repeat) -> modflow(prob, i, initValues, diffts) )
    ensSol = solve(ensemble; trajectories = N).u

    return sum( [sum((s ./ w).^2) for s in (ensSol .- y.u[2:end])] )
end

function trajdiff(w, y, npode, ::globalcost)
    N = length(y.t)-1 # should be optimized

    diffts = diff(y.t);
    initValues = y.u[1:end-1]
    # initial values don't matter, will be overwritten anyway
    x0 = y.u[1]
    tspan = (y.t[1], y.t[end])
    gpprob = ODEProblem(gpode!,x0, tspan, npode)
    gpSol = solve(gpprob; saveat = y.t).u

    return sum( [sum((s ./ w).^2) for s in (gpSol .- y.u)] )
end



function loglikelihood(w,y,npode, costtype)
    N = length(y.t)-1
    term3 = trajdiff(w, y, npode, costtype)
    vU = vec(reduce(hcat, npode.U))

    return -1/2 * vU' * (npode.kernel.Kchol \ vU) - 1/2 * log(det(npode.kernel.Kchol)) - 1/2 * term3 - N * sum(log.(w))
end

function detailloglikelihood(w,y,npode)
    N = length(y.t)-1
    term31 = trajdiff(w, y, npode, localcost())
    term32 = trajdiff(w, y, npode, globalcost())
    vU = vec(reduce(hcat, npode.U))

    t1 = -1/2 * vU' * (npode.kernel.Kchol \ vU) 
    t2 = - 1/2 * log(det(npode.kernel.Kchol)) 
    t31 = - 1/2 * term31 
    t32 = - 1/2 * term32
    t4 = - N * sum(log.(w))

    total = t1 + t2 + t4

    println("u'Ku: $t1")
    println("log det K: $t2")
    println("sum log w: $t4")
    println("local error:  $t31, total: $(t4 + t31)")
    println("global error: $t32, total: $(t4 + t32)")
    
end

function cost(x, y, ker, cost::T; sigma, U, w) where T <: costtype
    (sigma, nSigma) = getsigma(sigma, x)
    (w, nW) = getomega(w, x)
    (U, nU) = getu(U, nSigma, x)
    
    if length(x) != (nSigma + nW + nU)
        error("Arguments not consistent")
    end
    
    if any(sigma .< 0.001) | any(w .< 0)
        return -1000
    end
    
    ker = npODEs.kernel(ker.Z, typeof(ker.kerneltype)(sigma))
    npode = npODE(U, ker)
    return loglikelihood(w, y, npode, cost)
end



######################
## Aux functions
function modflow(prob, i, initValues, diffts)
    return DifferentialEquations.remake(prob; tspan = diffts[i], u0 = initValues[i])
end


function getsigma(sigma::Int, x)
    return (x[1:sigma], sigma)
end
function getsigma(sigma::Array{Float64, 1}, x)
    return (sigma, 0)
end

function getomega(w::Int, x)
    return (x[end-w+1:end], w)
end
function getomega(w::Array{Float64, 1}, x)
    return (w, 0)
end

function getu(U::Int, nSigma::Int, x)
    return (x[nSigma+1:nSigma+U], U)
end
function getu(U::Array{Float64, 1}, nSigma, x)
    return (U, 0)
end
