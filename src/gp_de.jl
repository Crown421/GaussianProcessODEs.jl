# https://github.com/SciML/DiffEqFlux.jl/blob/c59971fd4d3ee84aff39f88b7073d7e8cf51c34c/src/neural_de.jl#L38

basic_tgrad(u,p,t) = zero(u)


struct GPODE{M,P,RE,T,A,K} <: NeuralDELayer
    model::M
    p::P
    tspan::T
    args::A
    kwargs::K

    function GPODE(model,tspan,args...;p = initial_params(model),kwargs...)
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,tspan,args,kwargs)
    end
end