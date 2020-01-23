module ExogenousStates

import QuantEcon: MarkovChain

function MarkovChain(mc1::MarkovChain, mc2::MarkovChain, names=(:s1, :s2))
  
  combined_grid = [NamedTuple{names}((s1, s2)) for s1 in mc1.state_values for s2 in mc2.state_values]

  MarkovChain(kron(mc1.p, mc2.p), combined_grid)
end


export MarkovChain

end  

