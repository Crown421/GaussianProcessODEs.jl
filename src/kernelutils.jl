#####
# create new kernel with updated parameters
# currently limited functions
#####
function (ker::ScaledKernel)(w)
    typeof(ker)( ker.kernel(w[2:end]),  [w[1]])
end

# should include a check that w is as long as the current parametera
function (ker::TransformedKernel)(w)
    # not sure I need this, (eventually) just used for optimization
    l = 1/sqrt(2.0) ./ w 
#     l = w
    typeof(ker)(ker.kernel, typeof(ker.transform)(l))
end

# ToDO: eventually might need some getfield magic, for recursion. Problem is to know when to stop ( could probably dispatch on SimpleKernel (doesn't have parameters))
# SqExponentialKernel <: KernelFunctions.SimpleKernel > true
# but also, might not want to change the parameters in say gamma exponential kernel


#####
# Obtain scalar kernel parameters
#####
function getparam(x::Array{T, 1}) where {T <: Real}
     return x
end
function getparam(ker::TransformedKernel{K, ARDTransform{Array{Float64,1}}}) where K <: Kernel
    return 1/sqrt(2.) ./ ker.transform.v
end

function getparam(mker::K) where K <: npODEs.MatrixKernel
    ker = mker.kernel
    res = getparam.(getfield.(Ref(ker), fieldnames(typeof(ker))))
    return reduce(vcat, reverse(res))
end




#######
# compute kernel matrices
#######


function computeK(Z, kernelT::S) where {S <:matrixkernel }
    tmp = pairwise((z1, z2) -> kernelfunctionf(z1, z2, kernelT), Z,  Symmetric)
    K = symblockreduce(tmp)
end

function computeK(a, b, kernelT::S) where {S <:matrixkernel }
    tmp = [npODEs.kernelfunctionf(z1, z2, kernelT) for z1 in a, z2 in b]
    K = blockreduce(tmp)
end

function computeK(Z, kernelT::S) where {S <: scalarkernel}
    tmp = pairwise((z1, z2) -> kernelfunctionf(z1, z2, kernelT), Z,  Symmetric)
    x = Z[1]
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
    kron(tmp, Id)
end

function computeK(a, b, kernelT::S) where {S <: scalarkernel}
    tmp = [npODEs.kernelfunctionf(z1, z2, kernelT) for z1 in a, z2 in b]
    x = a[1]
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
    kron(tmp, Id)
end

#TODO add matrix valued version
# might have to take the symblock approach, allocate whole matrix and fill block by block

function computeK(a, kernelT)
    computeK(a, a, kernelT)
end

function Kx(x, Z, ker::T) where T <: scalarkernel
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
    scalars = kernelfunctionf.(Ref(x), Z, Ref(ker))
    (kron(scalars', Id))
end

function Kx(x, Z, ker::T) where T <: matrixkernel
    computeK([x], Z, ker)
end


function dKx(x, Z, ker::T) where T <: scalarkernel
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))

    tmp = npODEs.derivativekernelfunctionf.(Ref(x), Z, Ref(ker))
    tmp = reduce(hcat, tmp)
    (kron(tmp, Id))
end


###
# reduce matrix of arrays into blockmatrix
###

# export symblockreduce
# this one might/should be more efficient due to 
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

function blockreduce(A)
    reduce(vcat, [reduce(hcat, A[i, :]) for i in 1:size(A, 1)])
end