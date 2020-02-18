module Aiyagari

using QuantEcon
using Interpolations
using Optim
using Parameters
using LinearAlgebra: norm
using SparseArrays: sparse
using Arpack: eigs
using StructArrays
using ProgressMeter

include("Expectations.jl")
using .Expectations

abstract type Household end
struct Consumer <: Household end
struct Owner <: Household end
struct Renter <: Household end

include("bellman.jl")
include("stationary-distribution.jl")
include("aggregate-state.jl")


export solve_bellman
export controlled_markov_chain, stationary_distribution
export AggregateState
export Household, Owner, Renter, Consumer

end # module
