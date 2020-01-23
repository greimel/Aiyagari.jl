using Test, Aiyagari
using Optim, QuantEcon, Parameters, Interpolations

a_grid = LinRange(-2, 5, 20)

# Exogenous states (incomes)
z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid)

u(c) = c > 0 ? log(c) : 10000 * c - 10000 * one(c)

function Aiyagari.iterate_bellman!(value_new, value_old, policy, a_grid, z_mc, converged)
  r, β = 0.05, 0.9
  a_min, a_max = extrema(a_grid)
  
  for (i_z, z) in enumerate(z_mc.state_values)
    # Create interpolated expected value function
    exp_value = value_old * z_mc.p[i_z,:]

    itp_exp_value = interpolate(exp_value, BSpline(Cubic(Line(OnGrid()))))
    sitp_exp_value = scale(itp_exp_value, a_grid)
    
    for (i_a, a) in enumerate(a_grid) 
      
      obj(a_next) = u(a + z - a_next/(1+r)) + β * sitp_exp_value(a_next)

      res = optimize(a_next -> - obj(a_next), a_min, a_max)

      policy[i_a, i_z]    = Optim.minimizer(res)
      value_new[i_a, i_z] = - Optim.minimum(res)
      converged[i_a, i_z] = Optim.converged(res)
    end
  end
end

@btime @unpack value, policy = solve_bellman(a_grid, z_MC)
# 22 ms

a_min, a_max = a_grid[[1;end]]
@assert all(a_min .< policy .< a_max)

using Plots
plot(a_grid, value)
plot(a_grid, policy)

using BenchmarkTools

let value = value, policy=policy, z_mc = z_MC
  lin_ind = LinearIndices(size(value))

  ngp_exo = length(z_mc.state_values)
  n = length(policy)
  len = n * ngp_exo * 2

  I = zeros(Int, len)
  J = zeros(Int, len)
  V = zeros(len)

  @btime Aiyagari.controlled_markov_chain!($I, $J, $V, $lin_ind, $z_mc, $a_grid, $policy)
end

@btime dist = stationary_distribution(z_MC, a_grid, policy)
#926 μs

using Plots
plot(a_grid, reshape(dist, size(value)))

