#
export expKernel, rotexpKernel
export computeK # should probably not be exported

export keplerKernel

abstract type kerneltype end
# abstract type abstractkronkernel <: kerneltype  end
# abstract type abstracmatrixkernel <: kerneltype end

# temporary
abstract type scalarkernel <: kerneltype  end
abstract type matrixkernel <: kerneltype end

abstract type MatrixKernel <: kerneltype end
abstract type KronMatrixKernel <: kerneltype  end


struct uncoupledMKernel{kmc, K, Qt} <: KronMatrixKernel
    kernels::K
    Q::Qt
end


###
# Core structure that is part of each component of the structured matrix kernel
###
struct GIMKernel{GKP, K <: KernelFunctions.Kernel } <: MatrixKernel
    kernel::K
    groupaction::Function
    # may need to be changed for more general group action?
    parameterinterval::Tuple{Float64,Float64}
    gkparams::GKP

    GIMKernel{T,K}(ker::K, grpa, psp, gkparams::T) where {T,K} = new(ker, grpa, psp, gkparams)

end

function GIMKernel(ker::K, grpa, parameterinterval, N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, parameterinterval...))
    integralKernelCore{typeof(gkparams), K}(ker, grpa, parameterinterval, gkparams)
end

function GIMKernel(ker::K, grpa, parameterinterval, N::Nothing) where K
    integralKernelCore{typeof(N), K}(ker, grpa, parameterinterval, N)
end



function (ik::GIMKernel)(z1, z2)
        grpa = ik.iKCore.groupaction
        gkp = ik.iKCore.gkparams 
        ker = ik.iKCore.kernel
    
        base = gkp.weights .* map(x->ker(z1, grpa(x) * z2), gkp.x)
        return sum(base .* grpa.(gkp.x))
        # return sum(ker.(Ref(z1), grpa.(gkp.x) .* [z2]) .* f.(gkp.x) .* gkp.weights)
end


###
# Specialized implementation for rotational equivariant kernel
struct rotKernel{K <: KernelFunctions.Kernel, GKP }  <: MatrixKernel
    kernel::K
    gkparams::GKP
end

function rotKernel(ker::K; N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
    rotKernel{K, typeof(gkparams)}(ker, gkparams)
end

# rot(phi) = [cos(phi) -sin(phi); sin(phi) cos(phi)]
function rot(phi) 
    c = cos(phi)
    s = sin(phi)
    [c -s; s c]
end

function (rk::rotKernel)(z1, z2)
    gkp = rk.gkparams 
    ker = rk.kernel

    base = map(x -> ker(z1, rot(x) * z2), gkp.x) .* gkp.weights 
    costerm = sum(base .* cos.(gkp.x) )
    sinterm = sum(base .* sin.(gkp.x) )
    [costerm -sinterm; sinterm costerm]
end

function (mker::npODEs.rotKernel)(w::Array{T, 1}) where {T <: Real}
    typeof(mker)(mker.kernel(w), mker.gkparams)
end



### Old stuff


struct expKernel <: scalarkernel 
    param::Array{Float64, 1}
end
function kernelfunctionf(z1, z2, ker::expKernel)
    d = length(z1)
    w = ker.param[2:d+1] .^2  #better without square , very odd, likely issue with forward differences
    return ker.param[1]^2 * exp(-1/2 * wsqeuclidean(z1, z2, 1.0 ./w))
end


struct rotexpKernel <: matrixkernel 
    param::Array{Float64, 1}
    phis::Array{Float64, 1}
    weights::Array{Float64, 1}

    function rotexpKernel(param; N = 30)
        phis, weights = gauss(N, 0, 2*pi)
        new(param, phis, weights)
    end
end


function rotexpIntegrand(phi, x1, x2, w)
    npODEs.kernelfunctionf(x1, rot(phi)* x2, expKernel(w))
end

function kernelfunctionf(z1, z2, ker::rotexpKernel)
    d = length(z1)
    w = ker.param
    # w = vcat(ker.param[1], ker.param[2:d+1] .^2)  
    # costerm, _ = quadgk(phi -> rotexpIntegrand(phi, z1, z2, w)* cos(phi), 0, 2*pi, rtol = 1e-6)
    # sinterm, _ = quadgk(phi -> rotexpIntegrand(phi, z1, z2, w)* sin(phi), 0, 2*pi, rtol = 1e-6)

    phi = ker.phis
    weights = ker.weights

    base = rotexpIntegrand.(phi, Ref(z1), Ref(z2), Ref(w)) .*weights

    costerm = sum(base .* cos.(phi) )
    sinterm = sum(base .* sin.(phi) )

    K = [costerm -sinterm; sinterm costerm]
end

# very ad-hoc, needs much better implementation
struct keplerKernel <: matrixkernel 
    param::Array{Float64, 1}
    phis::Array{Float64, 1}
    weights::Array{Float64, 1}

    function keplerKernel(param; N = 30)
        phis, weights = gauss(N, 0, 2*pi)
        new(param, phis, weights)
    end
end

function keplrotexpIntegrand(phi, x1, x2, w)
    x2r = vcat(rot(phi)*x2[1:2], rot(phi)*x2[3:4])
    npODEs.kernelfunctionf(x1, x2r, expKernel(w))
end
function kernelfunctionf(z1, z2, ker::keplerKernel)
    d = length(z1)
    w = ker.param
    # w = vcat(ker.param[1], ker.param[2:d+1] .^2)  

    phi = ker.phis
    weights = ker.weights

    base1 = keplrotexpIntegrand.(phi, Ref(z1), Ref(z2), Ref(w))
    # base2 = rotexpIntegrand.(phi, Ref(z1[3:4]), Ref(z2[3:4]), Ref(w))

    costerm1 = sum(base1 .* cos.(phi) .*weights)
    sinterm1 = sum(base1 .* sin.(phi) .*weights)

    # costerm2 = sum(base2 .* cos.(phi) .*weights)
    # sinterm2 = sum(base2 .* sin.(phi) .*weights)

    K = [costerm1 -sinterm1 0 0;
         sinterm1 costerm1 0 0;
         0 0 costerm1 -sinterm1;
         0 0 sinterm1 costerm1]
    # K = [0 0 costerm2 -sinterm2;
    #      0 0 sinterm2 costerm2;
    #      costerm1 -sinterm1 0 0;
    #      sinterm1 costerm1 0 0]
end

function dist(z1, z2, ker)
    d = length(z1)
    w = ker.param[2:d+1] .^2    
    L = diagm( ( 1.0 ./ w))
    x = z1 - z2
    ker.param[1]^2 * exp(-1/2 * x'*L*x )
end

function derivativekernelfunctionf(z1, z2, ker::expKernel)
    d = length(z1)
    return ker.param[1]^2 * exp(-1/2 * weuclidean(z1, z2, 1.0 ./ ker.param[2:d+1])^2) * ( (z1.-z2)./ ker.param[2:d+1])
end

# function matrixkernelfunction(z1::Array{Float64, 1}, z2::Array{Float64, 1}, ker::T) where T <: scalarkernel
#     Id = Matrix{Float64}(LinearAlgebra.I, length(z1), length(z2))
#     return kernelfunctionf(z1, z2, ker) * Id
# end
