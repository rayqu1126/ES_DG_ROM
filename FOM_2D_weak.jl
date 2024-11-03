using OrdinaryDiffEq, StartUpDG, Trixi
using StructArrays, StaticArrays, SparseArrays
using LinearAlgebra, Plots
epsilon = 0.001
N = 4
num_elements = 16

equations = CompressibleEulerEquations2D(1.4)
rd = RefElemData(Quad(), SBP(), N)
md = MeshData(uniform_mesh(Quad(), num_elements), rd)

Trixi.flux_ec(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D) = 
    Trixi.flux_ranocha(u_ll, u_rr, orientation, equations)

    
function rhs_element!(du, u, parameters, t)
    (; equations, rd, md, epsilon) = parameters
    (; Qr_skew, Qs_skew) = parameters
    (; uf, uP, interface_flux, nxy) = parameters.prealloc

    fill!(du, zero(eltype(du)))

    uf .= view(u, rd.Fmask, :)
    uP .= view(uf, md.mapP)

    for i in md.mapB
        rho, rho_v1, rho_v2, E = uP[i]
        uP[i] = SVector(rho, -rho_v1, -rho_v2, E)
    end

    # compute interface fluxes
    @. interface_flux = (flux_ranocha(uf, uP, nxy, equations)) * md.Jf  # - 0.5 * max_abs_speed_naive(uf, uP, nxy, equations) * (uP - uf)) * md.Jf
    StructArrays.foreachfield((out, x) -> mul!(out, rd.LIFT, x), du, interface_flux)
    
    # compute inviscid terms
    for e in 1:md.num_elements
        # volume terms 
        for i in axes(Qr_skew, 1)
            # x flux
            cols = rowvals(Qr_skew) 
            for id in nzrange(Qr_skew, i)
                j = cols[id]
                if i > j
                    Fx_ij = flux_ranocha(u[i, e], u[j, e], 1, equations)
                    Fy_ij = flux_ranocha(u[i, e], u[j, e], 2, equations)
                    Fr_ij = md.rxJ[1, e] * Fx_ij + md.ryJ[1, e] * Fy_ij 
                    QF_ij = Qr_skew[i, j] * Fr_ij
                    du[i, e] += QF_ij / rd.M[i, i]
                    du[j, e] -= QF_ij / rd.M[j, j]
                end
            end  
            # y flux
            cols = rowvals(Qs_skew) 
            for id in nzrange(Qs_skew, i)
                j = cols[id]
                if i > j
                    Fx_ij = flux_ranocha(u[i, e], u[j, e], 1, equations)
                    Fy_ij = flux_ranocha(u[i, e], u[j, e], 2, equations)
                    Fs_ij = md.sxJ[1, e] * Fx_ij + md.syJ[1, e] * Fy_ij 
                    QF_ij = Qs_skew[i, j] * Fs_ij
                    du[i, e] += QF_ij / rd.M[i, i]
                    du[j, e] -= QF_ij / rd.M[j, j]
                end
            end           
        end
    end

   @. du /= -epsilon

    # # compute viscous terms
    # dudx = (md.rxJ .* dudr + md.sxJ .* duds + rd.LIFT * (@. 0.5 * (uP - uf) * md.nxJ)) ./ md.J
    # dudy = (md.ryJ .* dudr + md.syJ .* duds + rd.LIFT * (@. 0.5 * (uP - uf) * md.nyJ)) ./ md.J
    (; dudr, duds, dudx, dudy) = parameters.prealloc
    uf .= view(u, rd.Fmask, :)
    uP .= view(uf, md.mapP)
    StructArrays.foreachfield((out, x) -> mul!(out, rd.Dr, x), dudr, u)
    StructArrays.foreachfield((out, x) -> mul!(out, rd.Ds, x), duds, u)

    # @batch per=thread for i in axes(u,2)
    #     @views dudr[:,i] = rd.Dr * u[:,i]
    #     @views duds[:,i] = rd.Ds * u[:,i]
    # end

    @. interface_flux = 0.5 * (uP - uf) * md.nxJ
    StructArrays.foreachfield((out, x) -> mul!(out, rd.LIFT, x), dudx, interface_flux)
    @. interface_flux = 0.5 * (uP - uf) * md.nyJ
    StructArrays.foreachfield((out, x) -> mul!(out, rd.LIFT, x), dudy, interface_flux)

    @. dudx += md.rxJ * dudr + md.sxJ * duds
    @. dudy += md.ryJ * dudr + md.syJ * duds
    @. dudx /= md.J
    @. dudy /= md.J

    # just store the normal component 
    (; dudn_f, dudn_P, sigma_r, sigma_s) = parameters.prealloc
    @. dudn_f = @views dudx[rd.Fmask, :] * md.nx + dudy[rd.Fmask, :] * md.ny
    @. dudn_P = @views dudn_f[md.mapP]

    for i in md.mapB
        dudn_P[i] = -dudn_P[i]
    end

    @. sigma_r = md.rxJ * dudx + md.ryJ * dudy
    @. sigma_s = md.sxJ * dudx + md.syJ * dudy

    # the version of mul! which accumulates into `du`
    StructArrays.foreachfield((out, x) -> mul!(out, rd.Dr, x, one(eltype(x)), one(eltype(x))), du, sigma_r)
    StructArrays.foreachfield((out, x) -> mul!(out, rd.Ds, x, one(eltype(x)), one(eltype(x))), du, sigma_s)

    @. interface_flux = -0.5 * ((dudn_P + dudn_f) * md.Jf)
    StructArrays.foreachfield((out, x) -> mul!(out, rd.LIFT, x, one(eltype(x)), one(eltype(x))), du, interface_flux)

   @. du *= epsilon / md.J

end

Qr = rd.M * rd.Dr 
Qs = rd.M * rd.Ds 
Qr_skew = droptol!(sparse((Qr - Qr')), 100 * eps())
Qs_skew = droptol!(sparse((Qs - Qs')), 100 * eps())

# Initial Condition
rho = @. 1 + 0.5 * exp(-25 * ((md.x+0.5)^2+(md.y+0.5)^2))
v1 = @. 0 * rho
v2 = @. 0 * rho
p = @. rho^equations.gamma

# primitive variables to conservative variables
q = StructVector{SVector{4, Float64}}((rho, v1, v2, p))
u = prim2cons.(q, equations)

# prealloc variables
uf = u[rd.Fmask, :]
uP = uf[md.mapP]
interface_flux = similar(uf)
nxy = SVector.(md.nx, md.ny)
dudr = rd.Dr * u
duds, dudx, dudy = ntuple(_ -> similar(dudr), 3)
dudn_f = @. dudx[rd.Fmask, :] * md.nx + dudy[rd.Fmask, :] * md.ny
dudn_P = dudn_f[md.mapP]
sigma_r, sigma_s = similar(u), similar(u)
prealloc = (; uf, uP, interface_flux, nxy, 
              dudr, duds, dudx, dudy, 
              dudn_f, dudn_P, sigma_r, sigma_s)

h = estimate_h(rd,md)
cache = (; equations, rd, md, Qr_skew, Qs_skew, epsilon, h, 
            prealloc);
tspan = (0.0, 1.0)

ode = ODEProblem(rhs_element!, u, tspan, cache)

# function eigen_est(integrator)
#     (; epsilon, h) = integrator.p
#     integrator.eigen_est = 500 * inv(h^2)
# end

sol = solve(ode, RK4(), saveat=LinRange(tspan[1], tspan[2], 100),  callback=Trixi.AliveCallback(alive_interval=10))

xp, yp = rd.Vp * md.x, rd.Vp * md.y
scatter(vec(xp), vec(yp), zcolor=vec(rd.Vp * reshape(StructArrays.component(sol.u[end], 1), :, md.num_elements)) , 
    markersize=2, markerstrokewidth=0, legend=false, clims=(0.8, 1.15), axis=([], false), ticks = false, aspect_ratio=:equal, colorbar=false)
