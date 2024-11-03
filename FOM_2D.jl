using OrdinaryDiffEq, StartUpDG, Trixi
using StructArrays, StaticArrays, SparseArrays
using LinearAlgebra, Plots

epsilon = 0.001

N = 4
num_elements = 32
equations = CompressibleEulerEquations2D(1.4)
rd = RefElemData(Quad(), SBP(), N)

md = MeshData(uniform_mesh(Quad(), num_elements), rd)
md = make_periodic(md)

Trixi.flux_ec(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D) = 
    Trixi.flux_ranocha(u_ll, u_rr, orientation, equations)

# size(u) = size(md.x)
function rhs!(du, u, parameters, t)
    (; equations, rd, md, epsilon) = parameters
    (; Qr_skew, Qs_skew) = parameters
    (; uf, uP, interface_flux, nxy) = parameters.prealloc

    fill!(du, zero(eltype(du)))

    uf .= view(u, rd.Fmask, :)
    uP .= view(uf, md.mapP)

    # compute interface fluxes
    @. interface_flux = (flux_ranocha(uf, uP, nxy, equations) ) * md.Jf 
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

    (; dudr, duds, dudx, dudy) = parameters.prealloc
    StructArrays.foreachfield((out, x) -> mul!(out, rd.Dr, x), dudr, u)
    StructArrays.foreachfield((out, x) -> mul!(out, rd.Ds, x), duds, u)

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
# rho = @. 1 + 0.5 * exp(-25 * (md.x^2+md.y^2))
# v1 = @. 0 * rho
# v2 = @. 0 * rho
# p = @. rho^equations.gamma

## rho(x, y, t) = rho(x - t, y, t)
# v1 = @. 1 + 0 * rho
# v2 = @. 0 * rho
# p = @. 1 + 0 * rho


# Kelvin-Helmholtz
ICalpha, ICsigma = 0.1, 0.1
rho = @. 1 + 1/(1+exp(-(md.y+0.5)/ICsigma^2)) - 1/(1+exp(-(md.y-0.5)/ICsigma^2))
v1 = @. 1/(1+exp(-(md.y+0.5)/ICsigma^2)) - 1/(1+exp(-(md.y-0.5)/ICsigma^2)) - 0.5
v2 = @. ICalpha * sin(2 * pi * md.x) * @. (exp(-(md.y+0.5)^2/ICsigma^2) - exp(-(md.y-0.5)^2/ICsigma^2))
p = @. 2.5 + 0 * rho

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
tspan = (0.0, 3.0) 

ode = ODEProblem(rhs!, u, tspan, cache)

# function eigen_est(integrator)
#     (; epsilon, h) = integrator.p
#     integrator.eigen_est = 500 * inv(h^2)
# end

sol = solve(ode, Tsit5(), saveat=LinRange(tspan[1], tspan[2], 400), reltol = 1e-8, abstol = 1e-8, callback=Trixi.AliveCallback(alive_interval=10))

xp, yp = rd.Vp * md.x, rd.Vp * md.y
scatter(vec(xp), vec(yp), zcolor=vec(rd.Vp * reshape(StructArrays.component(sol.u[end], 1), :, md.num_elements)) , 
    markersize=2, markerstrokewidth=0, legend=false, axis=([], false), ticks = false, aspect_ratio=:equal, clims=(0.95, 2.1), colorbar = false)