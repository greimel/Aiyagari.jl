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

include("expectations-new.jl")

abstract type HouseState end
struct IsState <: HouseState end
struct NoState <: HouseState end

abstract type Household end
abstract type CoupledHouseholds end

@with_kw struct Consumer{T} <: Household
  𝔼::T = Unconditional()
end

@with_kw struct Owner{T1,T2<:HouseState} <: Household
  𝔼::T1 = Unconditional()
  state::T2 = NoState()
end

@with_kw struct Renter{T} <: Household
  𝔼::T = Unconditional()
end

𝔼(hh::Household) = hh.𝔼

@with_kw struct OwnOrRent{O<:Owner,R<:Renter} <: CoupledHouseholds
  owner::O = Owner()
  renter::R = Renter()
end

households(chh::OwnOrRent) = [chh.owner, chh.renter]
Base.length(chh::CoupledHouseholds) = length(households(chh))
 
include("bellman.jl")
include("stationary-distribution.jl")
include("aggregate-state.jl")


export solve_bellman
export controlled_markov_chain, stationary_distribution
export AggregateState
export Household, Owner, Renter, Consumer, OwnOrRent
export HouseState, IsState, NoState

export MarkovChain

examples = ["huggett", "housing", "morestates"]

end # module
