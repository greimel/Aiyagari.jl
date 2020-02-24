using Revise
using Test, Aiyagari
using QuantEcon, Parameters, Interpolations
using Plots

function marginal_distribution(exo, var)
  size_exo = size(exo)
  p_reshaped = reshape(exo.mc.p, (size_exo..., size_exo...))
  
  names = keys(exo)
  integrate_dim_bool = names .!= var
  
  n_dim = length(names)
  
  int_dim = findall([integrate_dim_bool...])
  
  int_col = int_dim .+ n_dim
   
  sum_columns = dropdims(sum(p_reshaped, dims=int_col), dims=Tuple(int_col))
  #sum_columns[:,:,:,1]
  dropdims(mean(sum_columns, dims=int_dim), dims=Tuple(int_dim))
end

## (Conditional) expectations of value functions

function conditional_expected_value(value, exo, i_exo, condition)
  len_endo = size(value,1)
  
  value_reshaped = reshape(value, (len_endo, size(exo)...))
  
  v = value_reshaped[:, [k == condition[1] ? condition[2] : Colon() for k in keys(exo)]...]

  oth_dim = keys(exo) .!= condition[1]

  len_exo_other = prod(size(exo)[[oth_dim...]])

  p = exo.mc.p
  π_ = reshape(p[i_exo,:], size(exo))[[k == condition[1] ? condition[2] : Colon() for k in keys(exo)]...]
  
  ∑π = sum(π_)
  π_ = ∑π == 0 ? π_ : π_ / ∑π
   
  reshape(v, (len_endo, len_exo_other)) * vec(π_)
  
end

@testset "exogenous states" begin
  x1_grid = [0.5; 1.0; 1.5]
  x1_prob = [0.7 0.15 0.15;
             0.2 0.6 0.2;
             0.15 0.15 0.7]
  x1_MC = MarkovChain(x1_prob, x1_grid, :x1)

  x2_grid = [0.15; 1.1; 1.15; 3.0]
  x2_prob = [0.15 0.0 0.7  0.15;
             0.6  0.2 0.0  0.2;
             0.15 0.15 0.7 0.0;
             0.6  0.2 0.0  0.2]
  x2_MC = MarkovChain(x2_prob, x2_grid, :x2)

  x3_grid = [0.3; 0.6]
  x3_prob = [0.7 0.3;
             1.0 0.0]
  x3_MC = MarkovChain(x3_prob, x3_grid, :x3)

  exo = ExogenousStatespace([x1_MC, x2_MC, x3_MC])

  @testset "marginal distribution" begin
    @test all(marginal_distribution(exo, :x1) .≈ x1_prob)
    @test all(marginal_distribution(exo, :x2) .≈ x2_prob)
    @test all(marginal_distribution(exo, :x3) .≈ x3_prob)
  end
  
  a_grid = LinRange(5, 10, 50)
    
  my_val = (a, exo) -> begin
    c = a + exo.x1 - (exo.x2 == 0.6)
    #c > 0 ? log(c) : 1 * c - 1
  end
  
  value = my_val.(a_grid, permutedims(exo.grid))
  
  @testset "conditional expectation" begin
    for i_exo in [1; 7; 14; 20; 24]
      for (i_dim, cond_dim) in enumerate([:x1, :x2, :x3])
        oth_dim = findall(keys(exo) .!= cond_dim)
        π_sub = dropdims(sum(reshape(exo.mc.p[i_exo,:], size(exo)), dims=oth_dim), dims=Tuple(oth_dim))

        𝔼V = mapreduce(+, 1:size(exo)[i_dim]) do x
         conditional_expected_value(value, exo, i_exo, cond_dim => x) * π_sub[x]
        end
        
        @test all(𝔼V .≈ value * exo.mc.p[i_exo,:])
      end
    end
  end
end
