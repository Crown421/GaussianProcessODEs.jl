using NLopt

export evalgpode, gpode!, optimizekernelpar, computeinitU, optimizeU, variance


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



# TODO: transition to update kernel 
# (or at least benchmark, if updating is not much faster, make struct immutable again)
function optimizekernelpar(U, ker; w = 0.0)
    opt = Opt(:LN_COBYLA, 3)
    opt.lower_bounds = [0., 0., 0.]
    opt.xtol_rel = 1e-6

    opt.max_objective = (par, g) -> npODEs.llutheta(U, npODEs.kernel(par, ker.grid; kernelfun = expKernel()); w = w)
    opt.maxeval = 1000

    (minf,minx,ret) = optimize(opt, ones(3))
    numevals = opt.numevals # the number of function evaluations
    println("got $minf at $minx after $numevals iterations (returned $ret)")

    ker = npODEs.kernel(minx, ker.grid; kernelfun = expKernel())
end

# TODO eventually extend to irregular data (non-equal timesteps)
function computeinitU(dt, Y, grid, par = [1.0, 1.0, 1.0])
    tgrid = trajgrid(Y[1:end-1]);
    tker = npODEs.kernel(tgrid, expKernel(par));

    tU = diff(Y) ./ dt;
    tnpODE = npODE(tU, tker);

    initU = evalgpode.(grid.Z, Ref(tnpODE))
end

function trajectoryloss(initU, ker, x0, tspan, dt, Y)
    transnpODE = npODE(initU, ker)
    transprob = ODEProblem(gpode!,x0, tspan, transnpODE)
    transsol = solve(transprob; saveat = dt);
    
    sum(abs2.(reduce(hcat, transsol.u .- Y))) / length(Y)
end

function optimizeU(initU, ker, x0, tspan, dt, Y; maxeval = 1500)
    iU = reduce(vcat, initU)

    opt = Opt(:LN_COBYLA, length(iU))
    opt.xtol_rel = 1e-3

    opt.min_objective = (U, g) -> trajectoryloss(U, ker, x0, tspan, dt, Y)
    opt.maxeval = maxeval

    (minf,optU,ret) = optimize(opt, iU)
    numevals = opt.numevals # the number of function evaluations
    println("got $minf after $numevals iterations (returned $ret)")

    return optU
end

    