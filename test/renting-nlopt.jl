using NLopt
using ForwardDiff

function a_next(c, h, states, agg_state, 𝔼V, params, hh::Renter)
  @unpack ρ, r = agg_state
  a = states.a
  y = states.z
  
  a_next = (1+r) * (a + y - c - ρ * h)
end

function objective0(c, h, states, agg_state, 𝔼V, params, hh::Renter)
  @unpack β = params
  
  a_next_ = a_next(c, h, states, agg_state, 𝔼V, params, hh)
  
  uu = c > 0 && h > 0 ? u(c,h) : 10_000 * min(c,h) - 10_000
  
  uu + β * 𝔼V(a_next_)  
  
end

function constraint0(c, h, states, agg_state, 𝔼V, params, hh::Renter)
  @unpack r = agg_state
  
  a_next_ = a_next(c, h, states, agg_state, 𝔼V, params, hh)
  # - y_min / r <= a
  #(1+r) * m <= p * h * (1-δ) * θ
  - a_next_
end

function objective_nlopt(x::Vector, grad::Vector, args...)
  if length(grad) > 0
    ForwardDiff.gradient!(grad, XX -> objective0(XX[1], XX[2], args...), x)
  end
  objective0(x[1], x[2], args...)
end

function constraint_nlopt(x::Vector, grad::Vector, args...)
  if length(grad) > 0
    ForwardDiff.gradient!(grad, xx -> constraint0(xx[1], xx[2], args...), x)
  end
  constraint0(x[1], x[2], args...)
end



function Aiyagari.get_optimum(states, agg_state, 𝔼V, params, a_grid, hh::Renter)
  opt = Opt(:LD_MMA, 2)
  #opt = Opt(:LD_SLSQP, 2)
  lower_bounds!(opt, [eps(), eps()])
  upper_bounds!(opt, [+Inf, params.h_thres])
  
  xtol_rel!(opt, √eps())
  ftol_rel!(opt, eps())

  maxeval!(opt, 150)
  
  max_objective!(opt, (x,g) -> objective_nlopt(x, g, states, agg_state, 𝔼V, params, hh))
  inequality_constraint!(opt, (x,g) -> constraint_nlopt(x, g, states, agg_state, 𝔼V, params, hh), eps())
    
  guess = sum(states)/2
  (max_f, max_x, ret) = optimize(opt, [guess, min(guess / agg_state.ρ, params.h_thres)])

  val = max_f
  c, h = max_x
  w_ = a_next(c, h, states, agg_state, 𝔼V, params, hh)
  
  conv = ret in [:FTOL_REACHED, :XTOL_REACHED, :SUCCESS, :LOCALLY_SOLVED]
  
  pol_full = (c=c, h=h, w_next=w_, ret=ret, conv=conv, count= opt.numevals)
  pol = pol_full.w_next, pol_full.h                    
      
  (pol=pol, pol_full=pol_full, val=val, conv=all(conv))
end

