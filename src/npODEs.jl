module npODEs

using DifferentialEquations
using Distances
using VectorizedRoutines
VR = VectorizedRoutines
using LinearAlgebra

include("types.jl")
include("kernelfuns.jl")
include("utils.jl")
include("base.jl")
# include("visualization.jl")

end # module
