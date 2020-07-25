export kernel, npODE
export trajectory

abstract type gridtype end


mutable struct kernel{S<:kerneltype}
    Kchol::Cholesky{Float64,Array{Float64,2}}
    Z::Array{Array{Float64,1},1}
    kerneltype::S

    function kernel{S}(Z, kernelT::S) where {S <:kerneltype }
        K = computeK(Z, kernelT)

        sigman = kernelT.param[end]
        K += sigman * I

        # TODO parametrize, check, test
        # Id = Matrix{Float64}(LinearAlgebra.I, size(K)... ) * 0.04
        # K = K + Id
        Kchol = LinearAlgebra.cholesky(K)

        new{S}(Kchol, Z, kernelT)
    end
end

function kernel(Z, kernelfun::S) where {S <: kerneltype}
    return kernel{S}(Z, kernelfun)
end

# function updatekernel!(𝜃, ker)
#     pKernel(z1, z2) = kerf(z1, z2, 𝜃, ker.kernel)
#     K = reduce(vcat, [reduce(hcat, [pKernel(ker.grid.Z[i], ker.grid.Z[j]) for j in 1:length(ker.grid.Z)]) for i in 1:length(ker.grid.Z)])
#     ker.Kchol = LinearAlgebra.cholesky(K)
#     ker.Kx = x -> reduce(vcat, pKernel.(Ref(x), ker.grid.Z))
# end

# at some point, change to Array{,1} and no reshape
struct npODE{T<:kernel}
    U::Array{Float64,1}
    kernel::T
    KiU::Array{Float64,2}
    
    function npODE{T}(vU, kernel) where T<:kernel
        KiU = kernel.Kchol \ vU
        KiU = reshape(KiU, :, 1)
        new{T}(vU, kernel, KiU)
    end

    # function npODE{T}(vU::Array{Float64, 1}, kernel) where T<:kernel
    #     new{T}([zeros(2), zeros(2)], vU, kernel)
    # end
end

# make array of real ?
npODE(U::Array{Float64,1}, kernel::T) where T<:kernel = npODE{T}(U, kernel)

function npODE(U::Array{Array{Float64,1},1}, kernel::T) where T<:kernel
    vU = reduce(vcat, U)
    npODE{T}(vU, kernel)
end


### type to store trajectories that might not be from DiffEq
struct trajectory
    t::Array{Float64,1}
    u::Array{Array{Float64,1},1}

    function trajectory(sol, dt = 0.05)
        t = collect(sol.t[1]:dt:sol.t[end])
        u = sol(t).u 
        new(t,u)
    end
end


