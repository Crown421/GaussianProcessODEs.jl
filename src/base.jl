export evalgpode, gpode!, optimizekernelpar, optimizeU, variance
export radialgrid, rectanglegrid


# function rectanglegrid(borders, stepSizes)
#     tmp = [range(borders[i][1], borders[i][end], step = stepSizes[i]) for i in 1:length(borders)] 
#     Z = collect.(collect(Iterators.product(tmp...)))[:]
# end

function rectanglegrid(borders, npoints::Array{Int, 1})
    tmp = [range(borders[i][1], borders[i][end], length = npoints[i]) for i in 1:length(borders)]
    Z = collect.(collect(Iterators.product(tmp...)))[:]
end

function rectanglegrid(borders, nGrid::Int)
    maxstep = maximum(reduce(vcat, diff.(borders)./nGrid))
    npoints = ceil.(reduce(vcat, (diff.(borders)./maxstep)))
    npoints = Int.(max.(npoints, 1.0))

    return rectanglegrid(borders, npoints)
end


function radialgrid(radius, nGridPoints; origin = [0,0])
    r = range(0, 3, length = nGridPoints+1)[2:end]
    th = collect(range(0, 2*pi, length = nGridPoints+1))[1:end-1]
    Z = [ [r*cos(th), r*sin(th)] for th in th, r in r][:] .+ [origin]
end



# function Kx(x, npODE::npODE)
#     Z = npODE.kernel.Z
#     ker = npODE.kernel.kerneltype
#     Kx(x, Z, ker)
# end

# # # this gives us the vector field x^dot = f(x) = evalGPODE(x[, npODE])
# function evalgpode(x, npODE)
#     Kxe = Kx(x, npODE)
#     # this is effectively (K^-1 * U)^T * Kxe^T = f^T (then "untransposed" with [:])
#     return (Kxe * npODE.KiU)[:]
# end

# function gpode!(dx, x, npODE, t)
#     dx[:] = evalgpode(x, npODE)
# end

# function variance(x, npODE)
#     Kxe = GaussianProcessODEs.Kx(x, npODE)
#     I - Kxe' * (npODE.kernel.Kchol \ Kxe)
# end
    