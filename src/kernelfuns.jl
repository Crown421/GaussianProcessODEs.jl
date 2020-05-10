#
export expKernel

abstract type scalarkernel <: kerneltype  end
abstract type matrixkernel <: kerneltype end

struct expKernel <: scalarkernel 
    param::Array{Float64, 1}
end
function kernelfunctionf(z1, z2, ker::expKernel)
    d = length(z1)
    return ker.param[1]^2 * exp(-1/2 * weuclidean(z1, z2, 1.0 ./ker.param[2:d+1])^2)
end

function derivativekernelfunctionf(z1, z2, ker::expKernel)
    d = length(z1)
    return ker.param[1]^2 * exp(-1/2 * weuclidean(z1, z2, 1.0 ./ ker.param[2:d+1])^2) * ( (z1.-z2)./ ker.param[2:d+1])
end

function matrixkernelfunction(z1::Array{Float64, 1}, z2::Array{Float64, 1}, ker::T) where T <: scalarkernel
    Id = Matrix{Float64}(LinearAlgebra.I, length(z1), length(z2))
    return kernelfunctionf(z1, z2, ker) * Id
end

function Kx(x, Z, ker::T) where T <: scalarkernel
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
    scalars = kernelfunctionf.(Ref(x), Z, Ref(ker))
    (kron(scalars, Id))
end

function Kx(x, npODE::npODE)
    Z = npODE.kernel.Z
    ker = npODE.kernel.kerneltype
    Kx(x, Z, ker)
end

function dKx(x, Z, ker::T) where T <: scalarkernel
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))

    tmp = npODEs.derivativekernelfunctionf.(Ref(x), Z, Ref(ker))
    tmp = reduce(hcat, tmp)
    (kron(tmp, Id))
end

function dKx(x, npODE::npODE)
    Z = npODE.kernel.Z
    ker = npODE.kernel.kerneltype
    dKx(x, Z, ker)
end


# export symblockreduce
function symblockreduce(A)
    innerN = size(A[1], 1)
    outerN = size(A, 1)
    n = innerN* outerN
    B = Symmetric(Matrix{typeof(A[1][1])}(undef, n, n))
    # @inbounds
    for j in 1:outerN
        modj = 1 + (j-1) * innerN
        for i in 1:outerN
            modi = 1 + (i-1) * innerN
            B.data[modi:modi+innerN-1, modj:modj+innerN-1] .= A[i,j]
        end
    end
    return B
end