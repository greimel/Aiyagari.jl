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
using Reexport

include("Expectations.jl")
@reexport using .Expectations

include("ExogenousStates.jl")
@reexport using .ExogenousStates

abstract type Household end
struct Consumer <: Household end
struct Owner <: Household end
struct Renter <: Household end
struct OwnOrRent <: Household end

include("bellman.jl")
include("stationary-distribution.jl")
include("aggregate-state.jl")


export solve_bellman
export controlled_markov_chain, stationary_distribution
export AggregateState
export Household, Owner, Renter, Consumer, OwnOrRent

export MarkovChain

end # module
