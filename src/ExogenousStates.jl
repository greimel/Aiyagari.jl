module ExogenousStates

import QuantEcon: MarkovChain, stationary_distributions

struct ExogenousStatespace{T1,T2,T3}
  grid::T1 # grid
  mc::T2
  dist::T3 # stationary distribution
end

function ExogenousStatespace(mc1, mc2; names=(:s1, :s2))
  grid = [NamedTuple{names}((s1, s2)) for (s1,s2) in Iterators.product(mc1.state_values, mc2.state_values)]
  
  mc = MarkovChain(mc1, mc2, names=names)
  
  @assert all(vec(grid) .== mc.state_values)
  
  ExogenousStates(grid, mc, stationary_distributions(dist)[1])
end

named_grid(grid, name) = NamedTuple{(name,)}.(Tuple.(grid))

function add_name(mc::MarkovChain, name::Symbol)
  MarkovChain(mc.p, named_grid(mc.state_values, name))
end

function MarkovChain(p, state_values, name::Symbol)
  MarkovChain(p, named_grid(state_values, name))
end

## Vector of independent Markov chains x1, x2

function product(mc1::MarkovChain, mc2::MarkovChain)
  
  combined_grid = [merge(s1, s2) for s1 in mc1.state_values for s2 in mc2.state_values]

  MarkovChain(kron(mc1.p, mc2.p), combined_grid)
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


export product, ExogenousStatespace

end  

