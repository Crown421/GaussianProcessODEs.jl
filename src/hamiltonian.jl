export crotinvKernel, Kx

# TODO: combine with other group action (currently f(phi,x) not rot(phi)*x)
function Krot(phi) 
    c = cos(phi)
    s = sin(phi)
    [c -s 0 0; s c 0 0; 0 0 c -s; 0 0 s c]
end

### invariant Kernel
struct crotinvKernel{K <: KernelFunctions.Kernel, GKP }  <: npODEs.MatrixKernel
    kernel::K
    gkparams::GKP
end

# constructor
function crotinvKernel(ker::K; N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
#     gkparams = zip(gkparams.x, gkparams.weights)
#     gkparams = Iterators.product(gkparams, gkparams)
    crotinvKernel{K, typeof(gkparams)}(ker, gkparams)
end

function (crker::crotinvKernel)(x1::Array{T,1},x2::Array{T,1}) where T <: Real
    phis = crker.gkparams.x
    
    rx1 = Krot.(phis) .* [x1]
    rx2 = Krot.(phis) .* [x2]
    
    return crker(rx1, rx2)
end

function (crker::crotinvKernel)(rx1::Array{Array{T,1},1}, rx2::Array{Array{T,1},1}) where T <: Real
    w = crker.gkparams.weights
    ker = crker.kernel
    iterIdx = Iterators.product(1:length(w), 1:length(w))
    
    tmp = mapreduce(iI -> w[iI[1]] * ker(rx1[iI[1]], rx2[iI[2]]) * w[iI[2]], + , iterIdx)
    return tmp
end

struct Kx{K, rZ}
    rker::K
    rotZ::rZ
end

function Kx(ker::K, Z) where K <: npODEs.MatrixKernel
    phis = ker.gkparams.x
    tmpRots = Krot.(phis)
    rotZ = [tmpRots .* [z] for z in Z]
    
    Kx{K, typeof(rotZ)}(ker, rotZ)
end

function (Kx::Kx)()
    rZ = Kx.rotZ
#     w = Kx.rker.gkparams.weights
#     ker = Kx.rker.kernel
#     iterIdx = Iterators.product(1:length(w), 1:length(w))
    
    N = length(Kx.rotZ)
    K = zeros(N, N)
    
    Threads.@threads for i in 1:N
        for j in 1:i
            tmp = Kx.rker(rZ[i], rZ[j])
#             tmp = mapreduce(iI -> w[iI[1]] * ker(rZ[i][iI[1]], rZ[j][iI[2]]) * w[iI[2]], + , iterIdx)
            K[i,j] = K[j,i] = tmp
        end
    end
    return K
end

function (Kx::Kx)(x)
    rZ = Kx.rotZ
#     w = Kx.rker.gkparams.weights
    ker = Kx.rker.kernel
#     iterIdx = Iterators.product(1:length(w), 1:length(w))
    
    phis = Kx.rker.gkparams.x
    rx = Krot.(phis) .* [x]
    
    D = length(ker(x,x))
    
    N = length(Kx.rotZ)
    K = Zygote.Buffer([0.], D,N)
#     zeros(1, N)
    
#     Threads.@threads for i in 1:N
    for i in 1:N
#         tmp = mapreduce(iI -> w[iI[1]] * ker(rx[iI[1]], rZ[i][iI[2]]) * w[iI[2]], + , iterIdx)
        tmp = Kx.rker(rx, rZ[i])
        for (j, val) in enumerate(tmp)
            K[j,i] = val
        end
#         display(K[:, i])
    end
    return copy(K)
end



struct d2crotinvKernel{K <: KernelFunctions.Kernel, GKP }  <: npODEs.MatrixKernel
    kernel::K
    gkparams::GKP
end

function d2crotinvKernel(ker::K; N::Int) where K <: dKernel
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
    d2crotinvKernel{K, typeof(gkparams)}(ker, gkparams)
end

function (crker::d2crotinvKernel)(x1::Array{T,1},x2::Array{T,1}) where T <: Real
    phis = crker.gkparams.x
    R = npODEs.Krot.(phis)
    rx1 = R .* [x1]
    rx2 = R .* [x2]
    return crker(R, rx1, rx2)
end

function (crker::d2crotinvKernel)(R, rx1::Array{Array{T,1},1}, rx2::Array{Array{T,1},1}) where T <: Real
    w = crker.gkparams.weights
    dker = crker.kernel
    iterIdx = Iterators.product(1:length(w), 1:length(w))
#     
    tmp = mapreduce(iI -> w[iI[1]] * R[iI[1]]' * dker(rx1[iI[1]], rx2[iI[2]]) * w[iI[2]], + , iterIdx)
    return tmp
end