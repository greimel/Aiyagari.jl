using Revise
using Test, Aiyagari
using QuantEcon, Parameters, Interpolations
using Plots

# Exogenous states (incomes)
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

x123_MC = product(x1_MC, x2_MC, x3_MC)

size_exo = (2,4,3)
states_reshaped = reshape(x123_MC.state_values, size_exo)
states_reshaped_SA = StructArray(states_reshaped)


function marginal_distribution(mc, var)
  size_exo = reverse((3,4,2))
  p_reshaped = reshape(mc.p, (size_exo..., size_exo...))
  
  names = keys(mc.state_values[1])
  integrate_dim_bool = reverse(names) .!= var
  
  n_dim = length(names)
  
  int_dim = findall([integrate_dim_bool...])
  
  int_col = int_dim .+ n_dim
   
  sum_columns = dropdims(sum(p_reshaped, dims=int_col), dims=Tuple(int_col))
  #sum_columns[:,:,:,1]
  dropdims(mean(sum_columns, dims=int_dim), dims=Tuple(int_dim))
end

all(marginal_distribution(x123_MC, :x1) .≈ x1_prob)
all(marginal_distribution(x123_MC, :x2) .≈ x2_prob)
all(marginal_distribution(x123_MC, :x3) .≈ x3_prob)


sum_columns = dropdims(sum(p_reshaped, dims=[4,5]), dims=(4,5))
sum_columns[:,:,:,1]
dropdims(mean(sum_columns, dims=[1,2]), dims=(1,2))

states_reshaped[1,1,Colon()]

states_reshaped.x2[:,3,:]
using StructArrays
sum(states_reshaped, dims=[1,3])
#function MarkovChain
add_name(x2_MC, :x2)

# Moving shocks
move_grid = Symbol[:just_moved, :move]
move_prob = [0.7 0.3;
             1.0 0.0]
move_MC = MarkovChain(move_prob, move_grid)

exo_MC = MarkovChain(x1_MC, move_MC, (:z, :move))

size_exo_ss = (length(x1_grid), length(move_grid))
a_grid = LinRange(5, 10, 50)

function my_val(a, exo)
  c = a + exo.z - (exo.move == :move)
  #c > 0 ? log(c) : 1 * c - 1
end

value = my_val.(a_grid, permutedims(exo_MC.state_values))
value
value_reshaped = reshape(value, (length(a_grid), reverse(size_exo_ss)...))

value_reshaped[:,1,:]

i_state = (i_endo = 10, i_z = 3, i_move = 2)


cond_exp_value(i_move) = move_MC.p[i_move,1] * value_reshaped[:,1,:] + move_MC.p[i_move,2] * value_reshaped[:,2,:]

plot(cond_exp_value(1))
plot!(cond_exp_value(2))

cond_exp_value(1) .- cond_exp_value(2)

