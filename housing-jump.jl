u(c; γ=γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

V(w_next) = log(w_next)

prices = (r = 0.02, p = 2.0)
parameters = (δ = 0.2, ω = 0.9, β = 0.8)

using JuMP
using Ipopt
using Parameters

function my_model(state, prices, parameters)
  @unpack r, p = prices
  #@unpack w, y = state
  @unpack δ, β, ω = parameters

  model = Model()
  set_optimizer(model, Ipopt.Optimizer)

  register(model, :u, 2, u, autodiff=true)
  register(model, :V, 1, V, autodiff=true)

  @variable(model, c >= eps())
  @variable(model, h >= eps())  

  @NLparameter(model, w == state.w)
  @NLparameter(model, y == state.y)
  
  w_next = @NLexpression(model, w + y - c - p * h * (r + δ))
  m = @NLexpression(model, p * h + c - y - w)
  @NLconstraint(model, (1+r) * m <= p * h * (1-δ) * ω)
  @NLobjective(model, Max, u(c,h) + β * V(w_next) )
  

  model, w, y, w_next
end


function my_solve(model, w, y, w_next, state, prices, parameters; init=false)
  if init
    model, w, y, w_next = my_model(state, prices, parameters)
  else
    set_value(w, state.w)
    set_value(y, state.y)
  end
  
  optimize!(model)
    
  (c=value(model[:c]), h=value(model[:h]), w_next=value(w_next))
end

state = state1
state1 = (y = 1.5, w = 0.5)
state2 = (y = 1.5, w = 0.7)
state3 = (y = 1.5, w = 0.9)

using BenchmarkTools
mod, w, y, w_next = my_model(state1, prices, parameters)
@time begin
  sol1 = my_solve(mod, w, y, w_next, state1, prices, parameters, init=true)
  sol2 = my_solve(mod, w, y, w_next, state2, prices, parameters, init=true)
  sol3 = my_solve(mod, w, y, w_next, state3, prices, parameters, init=true)
end # 37 ms

@btime begin
  model1, w, y = my_model(state1, prices, parameters)

  sol1 = my_solve(model1, w, y, state1, prices, parameters)
  sol2 = my_solve(model1, w, y, state2, prices, parameters)
  sol3 = my_solve(model1, w, y, state3, prices, parameters)
end # 39 ms




using BenchmarkTools
@btime optimize!(model) # 11 ms (m)

termination_status(model)
objective_value(model)
JuMP.value(c)
JuMP.value(h)

