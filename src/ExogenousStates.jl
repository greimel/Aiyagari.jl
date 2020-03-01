module ExogenousStates

using Parameters
import QuantEcon: MarkovChain, stationary_distributions

abstract type StateSpace end

struct EndogenousStateSpace{T0,T1,T2,T3,T4} <: StateSpace
  grids::T0
  grid::T1 # grid
  indices::T2
  linear_indices::T3
  size::T4
end

struct ExogenousStateSpace{T0,T1,T2,T3,T4,T5,T6} <: StateSpace
  grids::T0
  grid::T1 # grid
  indices::T2
  linear_indices::T3
  size::T4
  mc::T5
  dist::T6 # stationary distribution
end

Base.size(ss::StateSpace) = ss.size
Base.length(ss::StateSpace) = prod(size(ss))
Base.keys(ss::StateSpace) = keys(ss.grid[1])
MarkovChain(exo::ExogenousStateSpace) = exo.mc
dimension(ss::StateSpace) = length(size(ss))
linear_indices(ss::StateSpace) = ss.linear_indices

function EndogenousStateSpace(grids_nt::NamedTuple)
  keys_ = keys(grids_nt)
  
  size_ = Tuple(length.(collect(grids_nt)))
  
  grid0 = Iterators.product(grids_nt...)
  gridNT = NamedTuple{keys_}.(grid0)
  
  indices0 = Iterators.product([1:n for n in size_]...)
  indicesNT = NamedTuple{keys_}.(indices0)
  
  EndogenousStateSpace(grids_nt, gridNT, indicesNT, LinearIndices(size_), size_)
   
end

using Test

@testset "Endogenous state space" begin
  a_min = 0.0
  h_min = eps()
  
  a = LinRange(a_min, 7.5, 10)
  h = LinRange(h_min, 5.0, 15)

  endo = EndogenousStateSpace((a=a, h=h))
  
  @test size(endo) == (10, 15)
  @test length(endo) == 10 * 15
  @test keys(endo) == (:a, :h)
  @test endo.grid[1,1] == (a=a_min, h=h_min)
  @test endo.grid[5,7] == (a=a[5], h=h[7])
  @test endo.indices[3,14] == (a=3, h=14)
end



function ExogenousStateSpace(vec_mc)
  grids = getproperty.(vec_mc, :state_values)
  size = Tuple(length.(grids))
    
  mc = product(vec_mc...)
  
  grid = mc.state_values
  
  indices0 = collect(Iterators.product([1:n for n in size]...))
  indicesNT = NamedTuple{keys(grid[1])}.(indices0)
  
  ExogenousStateSpace(grids, grid, indicesNT, LinearIndices(size), size, mc, stationary_distributions(mc)[1])
end

named_grid(grid, name) = NamedTuple{(name,)}.(Tuple.(Ref.(grid)))

function add_name(mc::MarkovChain, name::Symbol)
  MarkovChain(mc.p, named_grid(mc.state_values, name))
end

function MarkovChain(p, state_values, name::Symbol)
  MarkovChain(p, named_grid(state_values, name))
end

## Vector of independent Markov chains x1, x2

function product(mc1::MarkovChain, mc2::MarkovChain)
  
  combined_grid = [merge(s1, s2) for s2 in mc2.state_values for s1 in mc1.state_values]

  MarkovChain(kron(mc2.p, mc1.p), combined_grid)
end

function product(mc_vec...)
  #@show eltype(mc_vec)
  @assert eltype(mc_vec) <: MarkovChain
  if length(mc_vec) == 1
     return mc_vec[1]
  else
    mc12 = product(mc_vec[1], mc_vec[2])
  end
  if length(mc_vec) == 2
    return mc12
  else
    return product(mc12, mc_vec[3], mc_vec[4:end]...)
  end
end


export EndogenousStateSpace, ExogenousStateSpace, dimension, linear_indices

end

