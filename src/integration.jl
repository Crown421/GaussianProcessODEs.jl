export cholupdate, cholmod!, cholmod, kcholmod!, kmemoryGP

function cholupdate(chol, Bt, C)
    d = size(C, 1)
    Lhat = zeros(size(chol).+size(C))

    Bttilde = Bt/chol.U
    Ctilde = C - Bttilde * Bttilde'
    Ltilde = cholesky(Ctilde).L

    Lhat[1:size(chol,1), 1:size(chol,2)] = chol.L
    # do better with lower triangular?
    Lhat[end-size(Bttilde,1)+1:end, 1:size(Bttilde,2)] = Bttilde
    # Lhat[1:size(Bttilde,2), end-size(Bttilde,1)+1:end] = permutedims(Bttilde)
    Lhat[end-size(Ltilde,1)+1:end, end-size(Ltilde,1)+1:end] = Ltilde
    # Lhat = hcat(vcat(chol.L, Bttilde), vcat(zeros(size(Bttilde,2), size(Bttilde,1)), Ltilde))
    cholhat = Cholesky(Lhat, :L, 0)
end

# # TODO: combine Code, also improve, to change chol.factors (should make in place much easier)
# function cholupdate!(chol, Bt, C)
#     d = size(C, 1)
#     Lhat = zeros(size(chol).+size(C))

#     Bttilde = Bt/chol.U
#     Ctilde = C - Bttilde * Bttilde'
#     Ltilde = cholesky(Ctilde).L

#     Lhat[1:size(chol,1), 1:size(chol,2)] = chol.L
#     # do better with lower triangular?
#     Lhat[end-size(Bttilde,1)+1:end, 1:size(Bttilde,2)] = Bttilde
#     # Lhat[1:size(Bttilde,2), end-size(Bttilde,1)+1:end] = permutedims(Bttilde)
#     Lhat[end-size(Ltilde,1)+1:end, end-size(Ltilde,1)+1:end] = Ltilde
#     # Lhat = hcat(vcat(chol.L, Bttilde), vcat(zeros(size(Bttilde,2), size(Bttilde,1)), Ltilde))
#     chol.factors = Lhat
# end

function cholmod!(chol, c, k)
    # ToDo dimension check?
    L11 = chol.L[1:k-1, 1:k-1]
    l22 = chol.L[k, k]
    l32 = chol.L[k+1:end, k]
    L31 = chol.L[k+1:end, 1:k-1]
    L33 = chol.L[k+1:end, k+1:end]
    
    c12 = c[1:k-1]
    c22 = c[k]
    c32 = c[k+1:end]

    lb12 = L11 \ c12
    lb22 = sqrt(c22 - lb12'*lb12)
    lb32 = (c32 - L31*lb12)/lb22 #./ 1.0136849372920411

    w2 = deepcopy(lb32)
    w1 = deepcopy(l32)
    
    L33ch = Cholesky(deepcopy(L33), :L, 0)
    lowrankupdate!(L33ch, w1)
    lowrankdowndate!(L33ch, w2)
    
    if chol.uplo == 'L'
        chol.factors[k, 1:k-1] = lb12
        chol.factors[k,k] = lb22
        chol.factors[k+1:end,k] = lb32
        chol.factors[k+1:end, k+1:end] = L33ch.L  
    else
        chol.factors[1:k-1, k] = lb12'
        chol.factors[k,k] = lb22'
        chol.factors[k, k+1:end] = lb32'
        chol.factors[k+1:end, k+1:end] = L33ch.L' 
    end
    chol
end

cholmod(chol, c, k) = cholmod!(deepcopy(chol), c, k)

function kcholmod!(Kchol, Kx, l)
    d = size(Kx, 1)
    idx = [-1, 0] .+ d*l

    for i in 1:d
        cholmod!(Kchol, Kx[i, :], idx[i])
    end
end



## really need to organize code, fork out the more stable bits, and have a dev package

mutable struct kmemoryGP{K, T<:Real}
    kernel::K
    memsize::Int
    iter::Base.Iterators.Stateful{Base.Iterators.Cycle{UnitRange{Int64}},Union{Nothing, Tuple{Int64,Int64}}}
    baseZ::Array{Array{T, 1},1}
    Z::Array{Array{T, 1},1}
    baseU::Array{Array{T, 1},1}
    U::Array{Array{T, 1},1}
    BaseKchol::Cholesky{T,Array{T,2}}
    Kchol::Cholesky{T,Array{T,2}}
    σ_n::T
end

function kmemoryGP(sgp::npODEs.SparseGP; k = 1)
    Kchol, _ = computeParts(sgp)
    U = sgp.inP[2]
    Z = deepcopy(sgp.inP[1])
    σ_n = sgp.σ_n
    kernel = sgp.kernel
    iter = Iterators.Stateful(Iterators.cycle(1:k))
    kmemoryGP{typeof(kernel), typeof(σ_n)}(kernel, k, iter, Z, Z, U, U, Kchol, Kchol, σ_n)
end

function (mGP::kmemoryGP)(x)
    ker = mGP.kernel
    Z = mGP.Z
    Kchol = mGP.Kchol
    vU = reduce(vcat, mGP.U)
    Kx = kernelmatrix(ker, [x], Z)
    
    m = Kx*(Kchol \ vU)
    
    Kxx = kernelmatrix(ker, [x], [x]) # noise term might also be needed here? thoughts!
    std = sqrt.(diag(Kxx - Kx * (Kchol \ Kx')))
    s = randn(length(x)) .* std
    
    fx = (m .+ s)
    
    m = length(mGP.baseZ)
    if length(Z) < m + mGP.memsize
        mGP.U = vcat(mGP.U, [fx])
        mGP.Z = vcat(mGP.Z, [x])
        
        mGP.Kchol = npODEs.cholupdate(Kchol, Kx, Kxx + mGP.σ_n*I)        
    else
        l = m + popfirst!(mGP.iter)
        d = length(x)
        mGP.U[l] = fx
        mGP.Z[l] = x
        offset = collect(-d+1:0)
        idx = offset .+ d*l
        Kx[:, idx] = Kxx + mGP.σ_n*I
        
        kcholmod!(mGP.Kchol, Kx, l)
        
    end
    
#     K = kernelmatrix(ker, mGP.Z) + mGP.σ_n * I
#     mGP.Kchol = cholesky(K)
    
    # TODO: think about the noise term here
#     d = length(x) #[:, end-d+1:end]
#     Kx2 = kernelmatrix(ker, [x], mGP.baseZ)
#     mGP.Kchol = npODEs.cholupdate(mGP.BaseKchol, Kx2, Kxx + mGP.σ_n*I)

    return fx
end





function computeParts(sgp)
    return _computeParts(sgp, sgp.inP, sgp.method)
end

function _computeParts(sgp, indP::NTuple{2, Array{<:Array{<:Real,1},1}}, method::M) where M
    Z = indP[1]
    U = indP[2]
    σ_n = sgp.σ_n
    ker = sgp.kernel
    vU = reduce(vcat, U)
    
    K = kernelmatrix(ker, Z) + σ_n * I
    Kchol = cholesky(K)
    return Kchol, vU
end