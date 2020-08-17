module npODEs

using DifferentialEquations
using Zygote
using Distances
using VectorizedRoutines: pairwise
using LinearAlgebra
using Optim
using QuadGK
using KernelFunctions

include("kernelfuns.jl")
include("types.jl")
include("kernelutils.jl")

include("opt.jl")
include("base.jl")
include("problems.jl")
include("utils.jl")
# include("visualization.jl")

end # module
