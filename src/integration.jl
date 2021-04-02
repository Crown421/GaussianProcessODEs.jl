# export cholupdate, cholmod!, cholmod, kcholmod!, kmemoryGP





# ## really need to organize code, fork out the more stable bits, and have a dev package

# mutable struct kmemoryGP{K, T<:Real}
#     kernel::K
#     memsize::Int
#     iter::Base.Iterators.Stateful{Base.Iterators.Cycle{UnitRange{Int64}},Union{Nothing, Tuple{Int64,Int64}}}
#     baseZ::Array{Array{T, 1},1}
#     Z::Array{Array{T, 1},1}
#     baseU::Array{Array{T, 1},1}
#     U::Array{Array{T, 1},1}
#     BaseKchol::Cholesky{T,Array{T,2}}
#     Kchol::Cholesky{T,Array{T,2}}
#     σ_n::T
# end

# function kmemoryGP(sgp::GaussianProcessODEs.SparseGP; k = 1)
#     Kchol, _ = computeParts(sgp)
#     U = sgp.inP[2]
#     Z = deepcopy(sgp.inP[1])
#     σ_n = sgp.σ_n
#     kernel = sgp.kernel
#     iter = Iterators.Stateful(Iterators.cycle(1:k))
#     kmemoryGP{typeof(kernel), typeof(σ_n)}(kernel, k, iter, Z, Z, U, U, Kchol, Kchol, σ_n)
# end

# function (mGP::kmemoryGP)(x)
#     ker = mGP.kernel
#     Z = mGP.Z
#     Kchol = mGP.Kchol
#     vU = reduce(vcat, mGP.U)
#     Kx = kernelmatrix(ker, [x], Z)
    
#     m = Kx*(Kchol \ vU)
    
#     Kxx = kernelmatrix(ker, [x], [x]) # noise term might also be needed here? thoughts!
#     std = sqrt.(diag(Kxx - Kx * (Kchol \ Kx')))
#     s = randn(length(x)) .* std
    
#     fx = (m .+ s)
    
#     m = length(mGP.baseZ)
#     if length(Z) < m + mGP.memsize
#         mGP.U = vcat(mGP.U, [fx])
#         mGP.Z = vcat(mGP.Z, [x])
        
#         mGP.Kchol = GaussianProcessODEs.cholupdate(Kchol, Kx, Kxx + mGP.σ_n*I)        
#     else
#         l = m + popfirst!(mGP.iter)
#         d = length(x)
#         mGP.U[l] = fx
#         mGP.Z[l] = x
#         offset = collect(-d+1:0)
#         idx = offset .+ d*l
#         Kx[:, idx] = Kxx + mGP.σ_n*I
        
#         kcholmod!(mGP.Kchol, Kx, l)
        
#     end
    
# #     K = kernelmatrix(ker, mGP.Z) + mGP.σ_n * I
# #     mGP.Kchol = cholesky(K)
    
#     # TODO: think about the noise term here
# #     d = length(x) #[:, end-d+1:end]
# #     Kx2 = kernelmatrix(ker, [x], mGP.baseZ)
# #     mGP.Kchol = GaussianProcessODEs.cholupdate(mGP.BaseKchol, Kx2, Kxx + mGP.σ_n*I)

#     return fx
# end





# function computeParts(sgp)
#     return _computeParts(sgp, sgp.inP, sgp.method)
# end

# function _computeParts(sgp, indP::NTuple{2, Array{<:Array{<:Real,1},1}}, method::M) where M
#     Z = indP[1]
#     U = indP[2]
#     σ_n = sgp.σ_n
#     ker = sgp.kernel
#     vU = reduce(vcat, U)
    
#     K = kernelmatrix(ker, Z) + σ_n * I
#     Kchol = cholesky(K)
#     return Kchol, vU
# end