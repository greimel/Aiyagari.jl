function u(c,h; ξ=0.8159, ρ=map(s -> (s-1)/s, 0.13), γ=2.0) # Housing share and Intra-temporal ES (0.13) from Garriga & Hedlund
  C = (ξ * h^ρ + (1-ξ) * c^ρ)^(1/ρ)
  u(C, γ=γ)
end

function transform_policies(plain_action, plain_state, prices, tech)
    @unpack h_next, w_next = plain_action
    @unpack y, w = plain_state
    @unpack p, r = prices
    @unpack θ, δ = tech
    κ₀ = (w+y)*(1+r) / (p * (1+r-θ))
    α₁ = h_next / κ₀
    κ₁ = (w+y) - p * h_next * δ
    κ₂ = p * h_next * (1-δ - θ)
    α₂ = (w_next - κ₂) / (κ₁ - κ₂)
    (α₁=α₁, α₂=α₂)
end

function get_back_policies(transf_action, plain_state, prices, tech)
    @unpack α₁, α₂ = transf_action
    @unpack y, w = plain_state
    @unpack p, r = prices
    @unpack θ, δ = tech
    κ₀ = (w+y)*(1+r) / (p * (1+r-θ))
    h_next = α₁ * κ₀

    κ₁ = (w+y) - p * h_next * δ
    κ₂ = p * h_next * (1-δ - θ)

    w_next = α₂ *(κ₁ - κ₂) + κ₂
    (w_next=w_next, h_next=h_next)
end

function obj(plain_action, plain_state, prices, tech, value_itp)
    @unpack h_next, w_next = plain_action
    @unpack y, w = plain_state
    @unpack p, r = prices
    @unpack θ, δ, β = tech

    h_next = h_next < eps() ? eps() : h_next
    m_next = (1-δ) * p * h_next - w_next
    c = w + y + m_next / (1+r) - p * h_next
    c > 0 ? u(c, h) + β * value_itp(w_next) : - 10000 + 10000 * c
end

function obj_bounded(α₁, α₂, plain_state, prices, tech, value_itp)
    plain_action = get_back_policies((α₁=α₁, α₂=α₂), plain_state, prices, tech)
    obj(plain_action, plain_state, prices, tech, value_itp)
end

function get_optimum(states, agg_state, 𝔼V, params, a_grid)
  plain_state = (w = states.a, y = states.z)
  lower = [eps(); eps()]
  upper = [1-eps(); 1-eps()]
  initial_x = [0.5; 0.5])
  
  res = optimize(α -> -obj_bounded(α[1], α[2], plain_state,
                     agg_state, params, 𝔼V),
                 lower, upper, initial_x, Fminbox(), autodiff=:forward)
                 
  converged[i] = any([res.f_converged; res.x_converged; res.g_converged])
  α₁, α₂ = res.minimizer
  transf_action = (α₁=α₁, α₂=α₂)
  plain_action =  get_back_policies(transf_action, plain_state, prices, tech)  

end

    