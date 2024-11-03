# ROM without hyper reduction

using Trixi
using NonNegLeastSquares
using LinearAlgebra

function get_POD_modes(Vsnap, N; weight_matrix = I)
    U, s = svd(weight_matrix * Vsnap)
    return weight_matrix \ U[:, 1:N], s
end

# construct V_snap
Vsnap_cons_vars = hcat([getindex.(hcat(sol.u...), i) for i in 1:nvariables(equations)]...)
Vsnap_entropy_vars = hcat([getindex.(cons2entropy.(hcat(sol.u...), equations), i) for i in 1:nvariables(equations)]...)
Vsnap = [Vsnap_cons_vars Vsnap_entropy_vars]

# set number of modes and get reduced basis
Nmodes = 30
weight_matrix = sqrt(M)
V_ROM, svd_values = get_POD_modes(Vsnap, Nmodes; weight_matrix)

# build ROM operators
M_ROM = V_ROM' * M * V_ROM
invM_ROM = inv(M_ROM)

VP_ROM = V_ROM * (M_ROM \ (V_ROM' * M))
invM_VTr = M_ROM \ V_ROM'

function rhs_ROM!(du_ROM, u_ROM, p, t)
    (; du, invM_VTr, V_ROM, VP_ROM, equations) = p
    u = entropy2cons.(VP_ROM * cons2entropy.(V_ROM * u_ROM, equations), equations)
    rhs!(du, u, p, t)
    du_ROM .= invM_VTr * M * du
    return du_ROM
end

params = (; ode.p..., 
            M_ROM, invM_VTr, V_ROM, VP_ROM, 
            equations, du = similar(ode.u0))
ode_ROM = ODEProblem(rhs_ROM!, pinv(V_ROM) * ode.u0, ode.tspan, params)
sol_ROM = solve(ode_ROM, RK4(), reltol = 1e-8, abstol = 1e-8,
            saveat=LinRange(tspan[1], tspan[2], 10), 
            callback=Trixi.AliveCallback(alive_interval=1000))

L2_error = sqrt(sum(M * norm.(sol.u[end] - V_ROM * sol_ROM.u[end]).^2)) / sqrt(sum(M*norm.(sol.u[end]).^2))