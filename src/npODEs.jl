module npODEs

using DifferentialEquations
using Distances
using VectorizedRoutines: pairwise
using LinearAlgebra
using Optim

include("kernelfuns.jl")
include("types.jl")

include("opt.jl")
include("base.jl")
include("problems.jl")
include("utils.jl")
# include("visualization.jl")

end # module
