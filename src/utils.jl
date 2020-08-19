export getborders
export gradient_observations, gradient_data

# TODO: possibly make smarter
function getborders(y)
    [[minimum(getindex.(y.u, 1))-0.2, maximum(getindex.(y.u, 1))+0.2], [minimum(getindex.(y.u, 2))-0.2, maximum(getindex.(y.u, 2))+0.2] ]
end

function _gradient_observations(sgp::SparseGP, indP::NTuple{2, Array{<:Array{<:Real,1},1}}) 
    X = indP[1]
    if length(X[1]) > 1
        error("gradient observations only available for 1D inputs")
    end
    Y = indP[2]
    X = reduce(vcat, X)
    ker = sgp.kernel
    σ_n = sgp.σ_n
    
    dker(t1,t2) = gradient(t1->ker(t1,t2), t1)[1]
    dK = dker.(X, permutedims(X))
    K = kernelmatrix(ker, X') + σ_n*I
    # Kchol = cholesky(K)

    D = dK/K
    tmp = reduce(hcat,Y)
    gpdode = D * permutedims(tmp)
    return [dx[:] for dx in eachrow(gpdode)]
end

function gradient_observations(sgp::SparseGP)
    return _gradient_observations(sgp, sgp.inP)
end


function gradient_data(traj, kernel = pskernel(ones(2)); show_opt = false)
    X = [ [x] for x in traj.t]
    Y = [ y for y in traj.u]
    
    traj_sgp = SparseGP(kernel, X, Y)
    traj_sgp = train_sparsegp(traj_sgp; show_opt = show_opt)
    
    dY = gradient_observations(traj_sgp)
    return (X=Y, Y=dY)
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