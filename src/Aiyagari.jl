module Aiyagari

using QuantEcon
using Interpolations
using Optim
using Parameters
using LinearAlgebra: norm
using SparseArrays: sparse
using Arpack: eigs

include("Expectations.jl")
using .Expectations

include("bellman.jl")
include("stationary-distribution.jl")

export solve_bellman
export controlled_markov_chain, stationary_distribution

end # module
