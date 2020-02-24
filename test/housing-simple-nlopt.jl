using NLopt
using ForwardDiff

function w_next(c, h, states, agg_state, 𝔼V, params, hh::Owner)
  @unpack p, r = agg_state
  @unpack δ = params
  @unpack z = states
  w = states.a
     
  w_next = w + z - c - p * h * (r + δ)
end

function m_next(c, h, states, agg_state, 𝔼V, params, hh::Owner)
  @unpack p = agg_state
  @unpack z = states
  w = states.a
  
  m = p * h + c - z - w
end

function objective0(c, h, states, agg_state, 𝔼V, params, hh::Owner)
  @unpack β = params
  
  w_next_ = w_next(c, h, states, agg_state, 𝔼V, params, hh::Owner)

  u(c,h) + β * 𝔼V([w_next_, w_next_, w_next_ - 0.4])    
end

function constraint0(c, h, states, agg_state, 𝔼V, params, hh::Owner)
  @unpack p, r = agg_state
  @unpack β, θ, δ = params
  
  m = m_next(c, h, states, agg_state, 𝔼V, params, hh::Owner)
  
  #(1+r) * m <= p * h * (1-δ) * θ
  (1+r) * m - (p * h * (1-δ) * θ)
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
  
function Aiyagari.get_optimum(states, agg_state, 𝔼V, params, a_grid, hh::Owner)
  @unpack h_thres = params
  
  # 1. check if feasible set is non-empty
  h_max = let
    w = states.a
    @unpack z = states
    @unpack δ, θ = params
    @unpack p, r = agg_state
    
    (w + z) / (p * (1 - (1-δ) * θ / (1+r)))
  end
  
  if h_max < h_thres
    conv = true
    pol_full = (c=NaN, h=NaN, m=NaN, w_next=NaN, ret=:infeasible, conv=conv, count=0)
    val = -Inf
    
  else
    opt = Opt(:LD_MMA, 2)
    #opt = Opt(:LD_SLSQP, 2)
    lower_bounds!(opt, [eps(), params.h_thres])
    
    xtol_rel!(opt, 1e-10)
    ftol_rel!(opt, 1e-10)
    
    maxeval!(opt, 250)
    
    max_objective!(opt, (x,g) -> objective_nlopt(x, g, states, agg_state, 𝔼V, params, hh))
    inequality_constraint!(opt, (x,g) -> constraint_nlopt(x, g, states, agg_state, 𝔼V, params, hh), 1e-8)
    
    guess = (states.a + states.z)/2
    #guesses = [guess, guess / agg_state.p]
    guesses = [guess/10, max(guess / agg_state.p, params.h_thres)]
    (max_f, max_x, ret) = optimize(opt, guesses)

    val = max_f
    c, h = max_x
    m = m_next(c, h, states, agg_state, 𝔼V, params, hh::Owner)
    w_ = w_next(c, h, states, agg_state, 𝔼V, params, hh::Owner)
    
    conv = ret in [:FTOL_REACHED, :XTOL_REACHED, :SUCCESS, :LOCALLY_SOLVED]
    
    pol_full = (c=c, h=h, m=m, w_next=w_, ret=ret, conv=conv, count= opt.numevals)
    
  end
  pol = pol_full.w_next, pol_full.h                    
      
  (pol=pol, pol_full=pol_full, val=val, conv= conv)
end

state1 =  (a = 0.01, z = 0.5)
grad1 = [0.0, 0.0]
#objective_nlopt([sum(state1)/2, sum(state1)/(2 * agg_state.p)], grad1, state1, agg_state, log, param)
#constraint_nlopt([sum(state1)/2, sum(state1)/(2 * agg_state.p)], grad1, state1, agg_state, log, param)
#@show grad1
