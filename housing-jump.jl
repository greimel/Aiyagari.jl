u(c; γ=γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

V(w_next) = log(w_next)

y, w, r, p = 1.5, 0.5, 0.02, 2.0
δ, ω, β = 0.2, 0.9, 0.8

using JuMP
using Ipopt

model0 = Model()
set_optimizer(model0, Ipopt.Optimizer)

register(model0, :u, 2, u, autodiff=true)
register(model0, :V, 1, V, autodiff=true)

@variable(model0, c >= eps())
@variable(model0, h >= eps())
@constraint(model0, (1+r) * p * h + c - y - w <= p * h * (1-δ) * ω)

@NLobjective(model0, Max, u(c,h) + β * V(w + y - c - p * h * (r + δ)) )

@btime optimize!(model0) # 10 ms (m)

value(c)
value(h)

model1 = Model()
set_optimizer(model1, Ipopt.Optimizer)

register(model1, :u, 2, u, autodiff=true)
register(model1, :V, 1, V, autodiff=true)

@variable(model1, c >= eps())
@variable(model1, h >= eps())

@variable(model1, w_next >= eps())
@variable(model1, m)
@constraint(model1, w_next == w + y - c - p * h * (r + δ))
@constraint(model1, m == p * h + c - y - w)
@constraint(model1, (1+r) * m <= p * h * (1-δ) * ω)
@NLobjective(model1, Max, u(c,h) + β * V(w_next) )

@btime optimize!(model1) # 24 ms (btime)

model2 = Model()
set_optimizer(model2, Ipopt.Optimizer)

register(model2, :u, 2, u, autodiff=true)
register(model2, :V, 1, V, autodiff=true)

@variable(model2, c >= eps())
@variable(model2, h >= eps())

w_next = @NLexpression(model2, w + y - c - p * h * (r + δ))
m = @expression(model2, p * h + c - y - w)
@constraint(model2, (1+r) * m <= p * h * (1-δ) * ω)
@NLobjective(model2, Max, u(c,h) + β * V(w_next) )

@btime optimize!(model2) # 11 ms (btime)

value(c)
value(h)

using BenchmarkTools
@btime optimize!(model) # 11 ms (m)

termination_status(model)
objective_value(model)
JuMP.value(c)
JuMP.value(h)

