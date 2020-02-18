using JuMP, Ipopt

function Aiyagari.get_optimum(states, agg_state, 𝔼V, params, a_grid)
  @unpack p, r = agg_state
  @unpack β, θ, δ = params
  
  model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
  
  MOI.set(model, MOI.RawParameter("print_level"), 0)
  #model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

  register(model, :u, 2, u, autodiff=true)
  register(model, :𝔼V, 1, w -> 𝔼V(w), autodiff=true)

  guess = sum(states)/2
    
  @variable(model, c >= eps(), start=guess)
  @variable(model, h >= eps(), start=guess/2)  

  @NLparameter(model, w == states.a)
  @NLparameter(model, y == states.z)
  
  w_next = @NLexpression(model, w + y - c - p * h * (r + δ))
  m = @NLexpression(model, p * h + c - y - w)
  @NLconstraint(model, (1+r) * m <= p * h * (1-δ) * θ)
  @NLobjective(model, Max, u(c,h) + β * 𝔼V(w_next) )
  
  MOI.set(model, MOI.Silent(), true)
  
  JuMP.optimize!(model)
  
  val = objective_value(model)

  pol_full = (c=value(c), h=value(h), m=value(m), w_next=value(w_next), conv=termination_status(model))
  pol = pol_full.w_next, pol_full.h                    
                  
  conv = termination_status(model) == 4
      
  (pol=pol, pol_full=pol_full, val=val, conv=conv)
end


# constrained optimization is provided in Optim!
#https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#nonlinear-constrained-optimization
