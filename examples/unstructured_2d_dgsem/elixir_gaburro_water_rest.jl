
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

equations = Gaburro2D(1.0, 2.78*10^5, 1000.0, 9.81)

function initial_condition_rest(x, t, equations::Gaburro2D)
    # liquid domain
    rho = equations.rho_0 * exp(-(equations.gravity * equations.rho_0/equations.k0) *(x[2] - 1.0))
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0
    
    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

initial_condition = initial_condition_rest

function boundary_condition_wallE(u_inner, normal_direction::AbstractVector, x, t,
    surface_flux_function, equations::Gaburro2D)
  
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction) 
  if normal_direction[2] == 0.0 # x direction
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
  else # y direction
    u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])
  end
  # Calculate boundary flux
  if (normal[1] > 0 || normal[2] > 0) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, normal_direction, equations)
  end
  return flux
end


boundary_condition = Dict( :Bottom  => boundary_condition_wallE,
                            :Right  => boundary_condition_wallE,
                            :Top  => boundary_condition_wallE,
                            :Left  => boundary_condition_wallE)


###############################################################################
# Get the DG approximation space

#volume_flux = (flux_central, flux_nonconservative_gaburro)
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)#(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 #volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the unstructured quad mesh from a file 
# create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "tank.mesh")

mesh = UnstructuredMesh2D(mesh_file)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition)

###############################################################################
# ODE solvers, callbacks, etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(stepsize_callback)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), abstol=1.0e-10, reltol=1.0e-10, dt=1.0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
