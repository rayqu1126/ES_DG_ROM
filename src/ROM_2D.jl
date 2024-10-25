# ROM without hyper reduction

using Trixi
using NonNegLeastSquares
using LinearAlgebra

FOM_dim = (N+1)^2 * md.num_elements
function get_POD_modes(Vsnap, N; weight_matrix = I)
    U, s = svd(weight_matrix * Vsnap)
    return weight_matrix \ U[:, 1:N], s
end

function Vprocess(Vsnap)
    Vsnap = hcat(StructArrays.components(Vsnap)...)
    return Vsnap
end

# construct global matrices if using elementwise
function rhs_x!(du, u, params, t)
    (; rd, md) = params
    uf .= view(u, rd.Fmask, :)
    uP .= view(uf, md.mapP)
    u_flux = @. 0.5 * (uP - uf) * md.nxJ
    du .= (md.rxJ .* (rd.Dr * u) + md.sxJ .* (rd.Ds * u) + rd.LIFT * u_flux) ./ md.J
end

function rhs_y!(du, u, params, t)
    (; rd, md) = params
    uf .= view(u, rd.Fmask, :)
    uP .= view(uf, md.mapP)
    u_flux = @. 0.5 * (uP - uf) * md.nyJ
    du .= (md.ryJ .* (rd.Dr * u) + md.syJ .* (rd.Ds * u) + rd.LIFT * u_flux) ./ md.J
end

u = similar(md.x)
du = similar(u)
u .= 0.0
du .= 0.0
uf = u[rd.Fmask, :]
uP = uf[md.mapP]

Ax = zeros(length(u), length(u))
Ay = zeros(size(Ax))
for i in eachindex(u)
    u[i] = 1
    rhs_x!(du, u, (;rd, md), 0.0)
    Ax[:, i] = vec(du)
    rhs_y!(du, u, (;rd, md), 0.0)
    Ay[:, i] = vec(du)
    u[i] = 0
end

M = kron(Diagonal(md.J[1,:]), rd.M)
Qx = sparse(M * Ax)
Qy = sparse(M * Ay)


Vsnap_cons_vars = Vprocess(reshape(hcat(sol.u...), FOM_dim,:))
Vsnap_entropy_vars = Vprocess(cons2entropy.(reshape(hcat(sol.u...), FOM_dim,:), equations))
Vsnap = [Vsnap_cons_vars Vsnap_entropy_vars]
Nmodes = 30
weight_matrix = sqrt(M)
V_ROM, svd_values = get_POD_modes(Vsnap, Nmodes; weight_matrix)

M_ROM = V_ROM' * M * V_ROM
invM_ROM = inv(M_ROM)

VP_ROM = V_ROM * (M_ROM \ (V_ROM' * M))
invM_VTr = M_ROM \ V_ROM'
invM_K_ROM = M_ROM \ (V_ROM' * Qx' * inv(M) * Qx * V_ROM) + M_ROM \ (V_ROM' * Qy' * inv(M) * Qy * V_ROM)


function rhs_ROM!(du_ROM, u_ROM, p, t)
    (; du, invM_VTr, V_ROM, VP_ROM, equations) = p
    u = entropy2cons.(VP_ROM * cons2entropy.(V_ROM * u_ROM, equations), equations)
    
    fill!(du, zero(eltype(du)))
    for i in axes(Qx, 1)
        for j in axes(Qx, 2)
            if i > j
                fij = flux_ec(u[i], u[j], 1, equations)
                QFij = 2 * Qx[i,j] * fij
                du[i] += QFij
                du[j] -= QFij
            end
        end
    end

    for i in axes(Qy, 1)
        for j in axes(Qy, 2)
            if i > j
                fij = flux_ec(u[i], u[j], 2, equations)
                QFij = 2 * Qy[i,j] * fij
                du[i] += QFij
                du[j] -= QFij
            end
        end
    end

    du_ROM .= - (invM_VTr * du + epsilon * invM_K_ROM * u_ROM)
    return du_ROM
end

params = (; ode.p..., 
            M_ROM, invM_VTr, V_ROM, VP_ROM, 
            equations, du = reshape(similar(ode.u0),FOM_dim,:))
ode_ROM = ODEProblem(rhs_ROM!, pinv(V_ROM) * reshape(ode.u0,FOM_dim,:) , ode.tspan, params)

## Don't run: 2D ROM (without HR) is very slow
# sol_ROM = solve(ode_ROM, Tsit5(),
#             saveat=LinRange(tspan[1], tspan[2], 10), 
#             callback=Trixi.AliveCallback(alive_interval=5))
