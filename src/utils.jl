export getborders, gridloglikelihood

# TODO: possibly make smarter
function getborders(y)
    [[minimum(getindex.(y.u, 1))-0.2, maximum(getindex.(y.u, 1))+0.2], [minimum(getindex.(y.u, 2))-0.2, maximum(getindex.(y.u, 2))+0.2] ]
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