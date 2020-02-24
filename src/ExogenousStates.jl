module ExogenousStates

using Parameters
import QuantEcon: MarkovChain, stationary_distributions

struct ExogenousStatespace{T1,T2,T3,T4,T5}
  grid::T1 # grid
  indices::T2
  size::T3
  mc::T4
  dist::T5 # stationary distribution
end

Base.size(exo::ExogenousStatespace) = exo.size
Base.length(exo::ExogenousStatespace) = prod(size(exo))
Base.keys(exo::ExogenousStatespace) = keys(exo.grid[1])
MarkovChain(exo::ExogenousStatespace) = exo.mc

function ExogenousStatespace(vec_mc)
  size = Tuple(length.(getproperty.(vec_mc, :state_values)))
    
  mc = product(vec_mc...)
  
  grid = mc.state_values
  
  indices0 = collect(Iterators.product([1:n for n in size]...))
  indicesNT = NamedTuple{keys(grid[1])}.(indices0)
  
  ExogenousStatespace(grid, indicesNT, size, mc, stationary_distributions(mc)[1])
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


export ExogenousStatespace

end  

