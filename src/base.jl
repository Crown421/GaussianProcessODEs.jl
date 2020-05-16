export evalgpode, gpode!, optimizekernelpar, optimizeU, variance


function llutheta(U, kernel, w)
    vU = vec(reduce(hcat, U))
    return -1/2 * vU' * (kernel.Kchol \ vU) - 1/2 * log(det(kernel.Kchol)) 
end

function llutheta(U, kernel)
    vU = vec(reduce(hcat, U))
    return -1/2 * vU' * (kernel.Kchol.U \ (kernel.Kchol.L \ vU)) - 1/2 * log(det(kernel.Kchol) + w) 
end

# # this gives us the vector field x^dot = f(x) = evalGPODE(x[, npODE])
function evalgpode(x, npODE)
    Kxe = Kx(x, npODE)
    # this is effectively (K^-1 * U)^T * Kxe^T = f^T (then "untransposed" with [:])
    return (npODE.KiU * Kxe)[:]
end

function gpode!(dx, x, npODE, t)
    dx[:] = evalgpode(x, npODE)
end

function evalJ(x, npODE)
    dKx = dKx(x, npODE)
    J = dKx * npODE.KiU'
    return reshape(J, length(x), length(x))
end

function evalR(x, npODE)
    Kx = Kx(x, npODE)
    return (Kx' / npODE.kernel.Kchol)[:]
end

function variance(x, npODE)
    Kxe = npODEs.Kx(x, npODE)
    I - Kxe' * (npODE.kernel.Kchol \ Kxe)
end
    