using npODEs
using Test
using BenchmarkTools

using KernelFunctions
using QuadGK

rot(phi) = npODEs.Krot(phi)

struct altrotinvKernel{K <: KernelFunctions.Kernel, GKP }  <: npODEs.MatrixKernel
    kernel::K
    gkparams::GKP
end

# constructor
function altrotinvKernel(ker::K; N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
    gkparams = zip(gkparams.x, gkparams.weights)
    gkparams = Iterators.product(gkparams, gkparams)
    altrotinvKernel{K, typeof(gkparams)}(ker, gkparams)
end


function (rk::rotinvKernel)(z1, z2)
    gkp = rk.gkparams 
    ker = rk.kernel

    mapreduce( g -> g[1][2]*ker(rot(g[1][1]) * z1, rot(g[2][1]) * z2)*g[2][2], +, gkp)
end

function (mker::rotinvKernel)(w::Array{T, 1}) where {T <: Real}
    typeof(mker)(mker.kernel(w), mker.gkparams)
end

import KernelFunctions: kernelmatrix
function kernelmatrix(ker::rotinvKernel, X)
    K = zeros(length(X), length(X))
    for i in 1:length(X), j in 1:i
        tmp = ker(X[i], X[j])
        K[i,j] = K[j,i] = tmp
    end
#     for (i, z1) in enumerate(X), (j,z2) in enumerate(X)
# #         K[i,j] = threadriker(ker, z1, z2)
#         K[i,j] = ker(z1,z2)
#     end
    return K
end



### setup

using LinearAlgebra
H(q,p) = norm(p)^2/2 - inv(norm(q))

borders = [[-2,2], [-2,2], [-2, 2], [-2, 2]]
ngp = 4
Z = rectanglegrid(borders, ngp) .+ [0.1*(rand(4) .- 0.5) for i in 1:ngp^length(borders)]
U = H.(getindex.(Z, Ref([1,2])), getindex.(Z, Ref([3,4])) );


### testing implementation above
@testset "invariance of reference" begin
    N = 55
    ker = pskernel(ones(5))
    riker = altrotinvKernel(ker; N);

    let phi = rand()*2pi
        @test (riker(Z[1], Z[2]) ≈ riker(rot(phi)*Z[1], Z[2]) ) atol = 1e-5
        @test (riker(Z[1], Z[2]) ≈ riker(Z[1], rot(phi)*Z[2]) ) atol = 1e-5
    end

end

### testing package implementation
@testset "invariance of package implementation" begin

    tmpZ = Z[1:60];
    criker = crotinvKernel(ker; N)

    let phi = rand()*2pi
        @test (criker(Z[1], Z[2]) ≈ criker(rot(phi)*Z[1], Z[2]) ) atol = 1e-5
        @test (criker(Z[1], Z[2]) ≈ criker(Z[1], rot(phi)*Z[2]) ) atol = 1e-5
    end

end

### comparison
let 
    println("Comparison")
    tmpZ = Z[1:60]
    println("reference")
    @time K1 = kernelmatrix(riker, tmpZ);
    K1 = kernelmatrix(riker, tmpZ);

    println("package")
    println("\r setup")
    @time tmpKx = Kx(criker, tmpZ);
    tmpKx = Kx(criker, tmpZ)
    println("\r kernel matrix")
    @time K2 = tmpKx();
    # @btime K2 = $tmpKx();
    K2 = tmpKx();

    @testset "accuracy" begin
        @test sum(K1 .- K2) < 1e-8
    end
end



### test package implementation


## complete prediction
let 
    tmpZ = Z[1:60]
    tmpKx = Kx(criker, tmpZ)

    x = rand(4)
    @time Kx1 = tmpKx(x);
    @time Kx2 = riker.([x], tmpZ)';
    @testset "Kx" begin
        @test sum(Kx1[:] .- Kx2[:]) < 1e-6
    end

    
end

let
@testset "Complete Prediction" begin 
    nZ = 6
    minZ = Z[1:nZ]
    minU = U[1:nZ]

    testker = pskernel([1., 12 .* ones(4)...])
    criker = crotinvKernel(testker; N)
    tmpKx2 = Kx(criker, minZ)
    K = tmpKx2()

    KinvU = K \ minU;
    # gpH(q,p) = (riker.([vcat(q,p)], permutedims(Zcl)) * KinvU)[1];
    gpf(z) = (tmpKx2(z) * KinvU)[1]

    @testset "recover data points" begin
        @test maximum(abs.(gpf.(minZ) .- minU)) < 1e-6
    end

    r1 = criker(Z[1], Z[2])
    phis = criker.gkparams.x
    rz1 = rot.(phis) .* [Z[1]]
    rz2 = rot.(phis) .* [Z[2]]

    r2 = criker(rz1, rz2)

    @testset "kernel itself" begin 
        @test r1 ≈ r2 atol=1e-6
    end
end

println("Done")
end

