export getborders
export gradient_observations, gradient_data


# TODO: possibly make smarter
# function getborders(y)
#     [[minimum(getindex.(y.u, 1))-0.2, maximum(getindex.(y.u, 1))+0.2], [minimum(getindex.(y.u, 2))-0.2, maximum(getindex.(y.u, 2))+0.2] ]
# end

function getborders(data::D) where D <: NamedTuple{(:X, :Y), <: NTuple{2,Array{<:Array{<:Real,1},1}}}
    borders = [[floor(minimum(getindex.(data.X, i))),
                    ceil(maximum(getindex.(data.X, i)))] 
                for i in 1:size(data.X[1],1)]
end

# function _gradient_observations(sgp::SparseGP, KiU, indP::NTuple{2, Array{<:Array{<:Real,1},1}}) 
function _gradient_observations(sgp::SparseGP, KiU, indP) 
    X = indP[1]
    X = reduce(vcat, X)
    ker = sgp.kernel
    σ_n = sgp.σ_n
    
    dker(t1,t2) = gradient(t1->ker(t1,t2), t1)[1]
    dK = dker.(X, permutedims(X))

    gpdode = dK * KiU
    dY = [dx[:] for dx in eachrow(gpdode)]
    return dY
end

function gradient_observations(gpm::GPmodel)
    sgp = gpm.sgp
    KiU = gpm.KinvU
    return _gradient_observations(sgp, KiU, sgp.inP)
end


"""
    gradient_data(traj, kernel = pskernel(ones(2)); show_opt = false)

Trains a kernel, builds a GPmodel, and then computes the gradients at the data points.

```julia-repl
traj_sgp = SparseGP(kernel, X, Y)
traj_sgp = train_sparsegp(traj_sgp; show_opt = show_opt)

gpmodel = GPmodel(traj_sgp)

dY = gradient_observations(gpmodel)
```
...

    gradient_data(traj, eX, kernel = pskernel(ones(2)); show_opt = false)

Trains a kernel, builds a GPmodel, and then computes output ands the gradients at the extended data points.
"""
function gradient_data(traj, kernel::K = pskernel(ones(2)); show_opt = false) where K<:Kernel
    X = [ [x] for x in traj.t]
    Y = [ y for y in traj.u]
    
    traj_sgp = SparseGP(kernel, X, Y)
    (dY,_) = _gradient_data(traj_sgp; show_opt)
    return (X=Y, Y=dY)
end

function gradient_data(traj, eX, kernel::K = pskernel(ones(2)); show_opt = false) where K<:Kernel
    X = [ [x] for x in traj.t]
    Y = [ y for y in traj.u]
    
    traj_sgp = SparseGP(kernel, eX, X, Y)
    (deY, gpmodel) = _gradient_data(traj_sgp; show_opt)
    eY = gpmodel.(eX)
    return (X=eY, Y=deY)
end

function _gradient_data(traj_sgp; show_opt)
    X = traj_sgp.inP[1]
    if length(X[1]) > 1
        error("gradient observations only available for 1D inputs")
    end

    traj_sgp = train_sparsegp(traj_sgp; show_opt = show_opt)

    gpmodel = GPmodel(traj_sgp)
    
    dY = gradient_observations(gpmodel)
    return (dY, gpmodel)
end









function dKx(x, npODE::npODE)
    Z = npODE.kernel.Z
    ker = npODE.kernel.kerneltype
    dKx(x, Z, ker)
end

function evalJ(x, npODE)
    devKx = dKx(x, npODE)
    J = devKx * npODE.KiU'
    return reshape(J, length(x), length(x))
end

function evalR(x, npODE)
    Kxm = Kx(x, npODE)
    return (Kxm' / npODE.kernel.Kchol)
end

function gridpointll(i, sigma, Z, U)
    id = collect(1:length(Z))
    deleteat!(id, i)
    Ztmp = Z[id]
    kertmp = npODEs.kernel(Ztmp, expKernel(sigma))
    Utmp = U[id]
    npVDPtmp = npODE(Utmp, kertmp)
    var = diag(variance(Z[i], npVDPtmp))
    predU = evalgpode(Z[i], npVDPtmp)
    err = predU - U[i]
    
    return -1/2* sum( (err./var).^2 ) - 1/2 * sum(log.(var))
end
function gridloglikelihood(sigma, Z, U)
    gridpointll.(1:length(Z), Ref(sigma), Ref(Z), Ref(U))
end