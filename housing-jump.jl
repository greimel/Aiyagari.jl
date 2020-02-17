u(c; γ=γ) = c^(1-γ) / (1-γ)

function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0)
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

V(w_next) = log(w_next)

using JuMP
using Ipopt

model = Model()
set_optimizer(model, Ipopt.Optimizer)

register(model, :u, 2, u, autodiff=true)
register(model, :V, 1, V, autodiff=true)
y, w, r, p = 1.5, 0.5, 0.02, 2.0
δ, ω, β = 0.2, 0.9, 0.8

@variable(model, c >= eps())
@variable(model, h >= eps())
@variable(model, w_next >= eps())
@variable(model, m)

@NLobjective(model, Max, u(c,h) + β * V(w_next) )
@constraint(model, w_next == w + y - c - p * h * (r + δ))
@constraint(model, (1+r) * m <= p * h * (1-δ) * ω)

@time optimize!(model)

termination_status(model)
objective_value(model)
JuMP.value(c)
JuMP.value(h)

