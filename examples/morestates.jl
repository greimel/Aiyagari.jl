# # Adding more states

using Revise #src
using Test #jl
using Aiyagari
using QuantEcon, Parameters
using Plots #md

PKG_HOME = joinpath(dirname(pathof(Aiyagari)), "..")

# ## Enlarging the state space

# First, let's add an additional exogenous state variable

z_grid = [0.5; 1.0; 1.5]
z_prob = [0.7 0.15 0.15;
          0.2 0.6 0.2;
          0.15 0.15 0.7]
z_MC = MarkovChain(z_prob, z_grid, :z)

x_grid = [1, 2, 3]
x_prob = [0.7 0.3 0.0;
          0.0 0.99 0.01;
          1.0 0.0 0.0]
          
x_MC = MarkovChain(x_prob, x_grid, :x)

exo1 = ExogenousStateSpace([z_MC])
exo2 = ExogenousStateSpace([z_MC, x_MC])

# Second, let's also add an additional endogenous state variable

w_grid = LinRange(0.0, 1.0, 40)
h_grid = LinRange(0.0, 2.6, 15)

endo1 = EndogenousStateSpace((w=w_grid, ))
endo2 = EndogenousStateSpace((w=w_grid, h=h_grid))

# ## Define some other things

include(joinpath(PKG_HOME, "test", "housing-simple-nlopt.jl"))

u(c; γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

mutable struct HousingAS{T1,T2,T3,T4} <: AggregateState
  r::T1
  p::T2 # house price
  ρ::T3 # rent
  dist::T4 # the distribution over idiosynchratic states
end

function HousingAS(r, p, endo, exo, param; ρ=p * (param.δ + r))
  dist_proto = zeros((length(endo), length(exo))) 
  HousingAS(r, p, ρ, dist_proto)
end

# ## Solve the model with and without redundant states

r = 0.15
p = 0.9
param  = (β = 0.7, θ = 0.9, δ = 0.1, h_thres = eps())
agg_state11 = HousingAS(r, p, endo1, exo1, param)
agg_state21 = HousingAS(r, p, endo2, exo1, param)
agg_state12 = HousingAS(r, p, endo1, exo2, param)
agg_state22 = HousingAS(r, p, endo2, exo2, param)

@time out11 = solve_bellman(endo1, exo1, agg_state11, param, Owner(state = NoState()), rtol=√eps())
#0.43 s (49 it)
@time out21 = solve_bellman(endo2, exo1, agg_state21, param, Owner(state = IsState()), rtol=√eps())
#4.85 s (49 it)
@time out12 = solve_bellman(endo1, exo2, agg_state12, param, Owner(state = NoState()), rtol=√eps())
#1.03 s (49 it)
@time out22 = solve_bellman(endo2, exo2, agg_state22, param, Owner(state = IsState()), rtol=√eps())
#14.41 s (49 it)

dist11 = stationary_distribution(endo1, exo1, out11.policies_full.w_next)
dist21 = stationary_distribution(endo2, exo1, out21.policy)
dist12 = stationary_distribution(endo1, exo2, out12.policies_full.w_next)
dist22 = stationary_distribution(endo2, exo2, out22.policy)

# Check if results are the same
val11 = out11.val
plt11 = plot(w_grid, val11, alpha=0.35, title="no redundant states") #md

plt21 = plot!(plt11, legend=false, title="redundant endogenous state") #md
@testset "irrelevant endogenous state" begin #jl
  val21 = reshape(out21.val, (size(endo2)..., length(exo1)))
  let #jl
    max_diff = Inf #jl
    for i_h in 1:length(h_grid)
      val_i = val21[:,i_h,:]
      plot!(plt21, w_grid, val_i, alpha = 0.35 / length(h_grid)) #md
      @test val11 ≈ val_i #jl
      max_i = maximum(abs, val11 .- val_i) #jl
      max_diff = max_diff < max_i ? max_diff : max_i  #jl
     end
    println(max_diff) #jl
  end #jl
end #jl


@testset "irrelevant exogenous state" begin #jl
  val12 = reshape(out12.val, (length(endo1), size(exo2)...)) #jl
  let #jl
    max_diff = Inf #jl
    for i_x in 1:length(x_grid) #jl
      val_i = val12[:,:,i_x] #jl
      @test val11 ≈ val_i #jl
      max_i = maximum(abs, val11 .- val_i) #jl
      max_diff = max_diff < max_i ? max_diff : max_i  #jl
    end #jl
    println(max_diff) #jl
  end #jl
end #jl
plt12 = plot!(plt11, w_grid, out12.val, alpha=0.35, legend=false,  title="redundant exogenous state") #md

plt22 = plot!(plt11, legend=false, title="redundant endo + exo state")  #md
@testset "irrelevant endogenous and exogenous states" begin #jl
   val22 = reshape(out22.val, (size(endo2)..., size(exo2)...))
   let #jl
     max_diff = Inf #jl
     for i_x in 1:length(x_grid)
       for i_h in 1:length(h_grid)
        val_i = val22[:,i_h,:,i_x]
        plot!(plt22, w_grid, val_i, alpha = 0.35 / length(x_grid) / length(h_grid)) #md
        @test val11 ≈ val_i #jl
        max_i = maximum(abs, val11 .- val_i) #jl
        max_diff = max_diff < max_i ? max_diff : max_i  #jl
      end
    end
    println(max_diff) #jl
  end #jl
end #jl

display(plt11) #md
#-
display(plt12) #md
#-
display(plt21) #md
#-
display(plt22) #md
#plot(plt11, plt12, plt21, plt22) #md

@testset "stationary distribution with redundant states" begin #jl
∑dist21 = dropdims(sum(reshape(dist21, (size(endo2)..., length(exo1))), dims=2), dims=2)
@test maximum(abs, ∑dist21 .- dist11) < 2e-6 #jl

∑dist12 = dropdims(sum(reshape(dist12, (length(endo1), size(exo2)...)), dims=3), dims=3)
@test maximum(abs, ∑dist12 .- dist11) < 2e-6 #jl

∑dist22 = dropdims(sum(reshape(dist22, (size(endo2)..., size(exo2)...)), dims=[2,4]), dims=(2,4))
@test maximum(abs, ∑dist22 .- dist11) < 2e-6 #jl
end #jl

plot(w_grid, dist11, legend=false) #md
plot!(w_grid, ∑dist21) #md
plot!(w_grid, ∑dist12) #md
plot!(w_grid, ∑dist22) #md
#- #md

# TODO: simple (linear or binary adjustment costs)
# TODO: house quality shock 