#
export expKernel

struct expKernel <: kerneltype end
# eventually, think about making parameters part of struct?

function kerf(z1, z2, 𝜃, ::expKernel)
    Id = Matrix{Float64}(LinearAlgebra.I, length(z1), length(z1))
    return 𝜃[1]^2 * exp(-1/2 * weuclidean(z1, z2, 𝜃[2:end])) * Id
end