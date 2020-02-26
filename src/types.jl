export radialgrid, rectgrid, trajgrid, kernel, updatekernel!, npODE

abstract type kerneltype end
abstract type gridtype end


# might almost make sense to make this a parametrized type (grid, with struct radial radius, origin,  and struct rect borders), 
# to ensure common fields (Z, nGridPoints) are not doubled, but its fine for now
# todo, also find better word for nGridPoints, when building the grid (only number of 1D intervals)
struct radialgrid <: gridtype 
    radius::Float64 
    origin::Array{Float64}
    Z::Array{Array{Float64,1},1}
    nGridPoints::Int

    function radialgrid(radius, nGridPoints; origin = [0,0])
        r = range(0, 3, length = nGridPoints+1)[2:end]
        th = collect(range(0, 2*pi, length = nGridPoints+1))[1:end-1]
        Z = [ [r*cos(th), r*sin(th)] for th in th, r in r][:] .+ [origin]
        nGridPoints = length(Z)
        new(radius, origin, Z, nGridPoints)
    end

end


struct rectgrid <: gridtype
    borders::Array{Array{Float64,1},1}
    Z::Array{Array{Float64, 1}, 1}
    nGridPoints::Int

    function rectgrid(borders, stepSizes)
        tmp = [range(borders[i][1], borders[i][end], step = stepSizes[i]) for i in 1:length(borders)] 
        Z = collect.(collect(Iterators.product(tmp...)))[:]
        nGridPoints = length(Z)
        new(borders, Z, nGridPoints)
    end
end

function rectgrid(borders, nGrid::Int)
    stepSizes = ones(length(borders))*maximum(reduce(vcat, diff.(borders)./nGrid))
    return rectgrid(borders, stepSizes)
end


struct trajgrid <: gridtype
    Z::Array{Array{Float64,1},1}
    nGridPoints::Int

    function trajgrid(Y)
        nG= length(Y)
        new(Y, nG)
    end
end



mutable struct kernel{T<:gridtype, S<:kerneltype}
    Kchol::Cholesky{Float64,Array{Float64,2}}
    Kx::Function
    grid::T
    kernel::S

    function kernel{T, S}(𝜃, grid::T, kernelT::S) where {T <: gridtype, S <:kerneltype }
        pKernel(z1, z2) = kerf(z1, z2, 𝜃, kernelT)
        K = reduce(vcat, [reduce(hcat, [pKernel(grid.Z[i], grid.Z[j]) for j in 1:length(grid.Z)]) for i in 1:length(grid.Z)])
        Kchol = LinearAlgebra.cholesky(K)
        Kx(x) = reduce(hcat, pKernel.(Ref(x), grid.Z))

        new{T,S}(Kchol, Kx, grid, kernelT)
    end
end

function kernel(𝜃, grid::T; kernelfun::S = expKernel) where {T<: gridtype, S <: kerneltype}
    return kernel{T,S}(𝜃, grid, kernelfun)
end

function updatekernel!(𝜃, ker)
    pKernel(z1, z2) = kerf(z1, z2, 𝜃, ker.kernel)
    K = reduce(vcat, [reduce(hcat, [pKernel(ker.grid.Z[i], ker.grid.Z[j]) for j in 1:length(ker.grid.Z)]) for i in 1:length(ker.grid.Z)])
    ker.Kchol = LinearAlgebra.cholesky(K)
    ker.Kx = x -> reduce(vcat, pKernel.(Ref(x), ker.grid.Z))
end


struct npODE{T<:kernel}
    U::Array{Array{Float64, 1}, 1}
    vU::Array{Float64,1}
    kernel::T
    
    function npODE{T}(U, kernel) where T<:kernel
        vU = reduce(vcat, U)
        new{T}(U, vU, kernel)
    end

    function npODE{T}(vU::Array{Float64, 1}, kernel) where T<:kernel
        new{T}([zeros(2), zeros(2)], vU, kernel)
    end
end

npODE(U, kernel::T) where T<:kernel = npODE{T}(U, kernel)


