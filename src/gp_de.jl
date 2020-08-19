export SparseGP, GPmodel, GPODE

# https://github.com/SciML/DiffEqFlux.jl/blob/c59971fd4d3ee84aff39f88b7073d7e8cf51c34c/src/neural_de.jl#L38


###
# Struct that contains everything needed for prediction
###
struct SparseGP{K, T<:Real, N, A<: NTuple{N, Array{<:Array{<:Real,1},1}}}
    kernel::K
    σ_n::T
    inP::A
    mean::Function
    trafo::Function
    # type? FITC, SOR, PITC
end

#ToDo: Update σ_n in this object

zeromean(x) = fill(0, size(x))
identitytrafo(x) = x

function SparseGP(kernel, Z, U; σ_n = 1e-6, mean = zeromean, trafo = identitytrafo)
    indP = (Z, U)
    N = length(indP)
    SparseGP{typeof(kernel), typeof(σ_n), N, typeof(indP)}(kernel, σ_n, indP, mean, trafo)
end
function SparseGP(kernel, Z, X, Y; σ_n = 1e-6, mean = zeromean, trafo = identitytrafo)
    indP = (trafo.(Z), trafo.(X), Y)
    N = length(indP)
    SparseGP{typeof(kernel), typeof(σ_n), N, typeof(indP)}(kernel, σ_n, indP, mean, trafo)
end

function (sgp::SparseGP)(x::T) where T <: Real
    Z = sgp.inP[1]
    ker = sgp.kernel
    Kx = kernelmatrix(ker, [[x]], Z)
end

function (sgp::SparseGP)(x::Array{T,1}) where T <: Real
    Z = sgp.inP[1]
    ker = sgp.kernel
    Kx = kernelmatrix(ker, [x], Z)
end



###
# GP model, that contains the sparse GP object with all necessary data, and 
struct GPmodel{SGP <: SparseGP, T<:Real}
    sgp::SGP
    KinvU::Array{T,2}
end 

# maybe with data
function GPmodel(sgp::SparseGP)
    KiU = computeKinvU(sgp)
    GPmodel(sgp, KiU)
end

function (gpm::GPmodel)(x)
    Kx = gpm.sgp(gpm.sgp.trafo(x))
    μ = gpm.sgp.mean
    return (μ(x) .+ (Kx * gpm.KinvU))[:]
end


####
# functions to facilitate efficient computation, as per Q-C&R
function computeKinvU(sgp::SparseGP)
    return _computeKinvU(sgp, sgp.inP)
end

function _computeKinvU(sgp::SparseGP, indP::NTuple{2, Array{<:Array{<:Real,1},1}}) 
    Z = indP[1]
    U = indP[2]
    σ_n = sgp.σ_n
    μ = sgp.mean
    U = U .- μ.(Z)
#     vU = reduce(vcat, U)
    # ToDo: might have to make output(?) dimensions more explicit
    vU = reshape(reduce(vcat, U), :, length(Z)*length(Z[1]))
    vU = permutedims(vU)
    ker = sgp.kernel
    K = kernelmatrix(ker, Z) + σ_n * I
    return K \ vU
end

function _computeKinvU(sgp::SparseGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}}) 
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    ker = sgp.kernel
    
    Kff = kernelmatrix(ker, X)
    Kfu = kernelmatrix(ker, X, Z)
    Kuu = kernelmatrix(ker, Z)
    
    Qff = Kfu * ( Kuu \ Kfu' )
    
    noise = sgp.σ_n
    Λ = diagm(diag( Kff - Qff) .+ noise)
    
    Σ = Kuu + Kfu' * (Λ \ Kfu)
    
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    vY = permutedims(vY)
    return Σ \ (Kfu' * (Λ \ vY))
end

#####
# complete GPODE construct
#####
basic_tgrad(u,p,t) = zero(u)
struct GPODE{M<:GPmodel,T,A,K} #<: NeuralDELayer
    model::M
    # p::P, maybe one day
    tspan::T
    args::A
    kwargs::K

    function GPODE(model::GPM,tspan,args...; kwargs...) where GPM <: GPmodel
        new{typeof(model),typeof(tspan),typeof(args),typeof(kwargs)}(
            model,tspan,args,kwargs)
    end
end

function GPODE(sgp::SGP,tspan,args...;kwargs...) where SGP <: SparseGP
    gpm = GPmodel(sgp)
    return GPODE(gpm,tspan,args...,kwargs...)
end

function (gp::GPODE)(x)
    dudt_(u,p,t) = gp.model(u)
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,getfield(gp,:tspan))
    solve(prob,gp.args...;gp.kwargs...)
end

