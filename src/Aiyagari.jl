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

include("expectations.jl")

abstract type Household end
@with_kw struct Consumer{T} <: Household
  𝔼::T = Unconditional()
end

@with_kw struct Owner{T} <: Household
  𝔼::T = Unconditional()
end

@with_kw struct Renter{T} <: Household
  𝔼::T = Unconditional()
end
@with_kw struct OwnOrRent{O<:Owner,R<:Renter} <: Household
  owner::O = Owner()
  renter::R = Renter()
end

𝔼(hh::Household) = hh.𝔼
 
include("bellman.jl")
include("stationary-distribution.jl")
include("aggregate-state.jl")


export solve_bellman
export controlled_markov_chain, stationary_distribution
export AggregateState
export Household, Owner, Renter, Consumer, OwnOrRent

export MarkovChain

examples = ["huggett", "housing"]

end # module
