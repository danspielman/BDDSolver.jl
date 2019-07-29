module BDDSolver

    using Laplacians
    using StaticArrays
    using Arpack
    using Statistics

    using SparseArrays
    using LinearAlgebra

    using Random   
    
    include("BlockCSC.jl")
    include("BDDTypes.jl")
    include("acBDD.jl")
    include("approxPQ.jl")
    include("generators.jl")
    include("solvers.jl")

    export approxCholBDD
    export BlockCSC

end # module
