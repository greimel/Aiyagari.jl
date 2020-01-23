module Expectations

using LinearAlgebra: dot
using QuantEcon: MarkovChain

export expected_value

expected_value(i_exo, i_endo_next, mc_exo, value) = dot(value[i_endo_next,:], mc_exo.p[i_exo,:])
expected_value(i_exo, mc_exo, value) = value * mc_exo.p[i_exo,:]

using Test
let
  V = [1 2 3;
       4 5 6;
       7 8 9;
       2 5 8]

  mc = MarkovChain([0.1 0.8 0.1;
                    0.8 0.1 0.1;
                    0.1 0.1 0.8])
  @testset "expected value" begin
    @test expected_value(1, 1, mc, V) == dot(V[1,:], mc.p[1,:]) == 2.0
    @test expected_value(2, 3, mc, V) == dot(V[3,:], mc.p[2,:]) ≈ 7.3
    @test expected_value(3, mc, V)[4] == expected_value(3, 4, mc, V) ≈ 7.1
  end
  
end

end
