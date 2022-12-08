
using OrdinaryDiffEq
using Trixi


equations = Gaburro2D(1.0, 1.0*10^3, 1000.0, 9.81)

function initial_condition_const(x, t, equations::Gaburro2D)

    # liquid domain
    if x[2] < 0.5
        rho = 50.5
    else
        rho = 50.0
    end 
    rho = 100.0
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0

    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end

function initial_condition_const_exp(x, t, equations::Gaburro2D)

    # liquid domain
    rho = 100 * exp(- x[2])
    v1 = 0.0
    v2 = 0.0
    alpha = 1.0

    return prim2cons(SVector(rho, v1, v2, alpha), equations)
end
  
initial_condition = initial_condition_const

boundary_condition = Dict( :B1  => boundary_condition_wall,
                            :B2  => boundary_condition_wall,
                            :B3  => boundary_condition_wall,
                            :B4  => boundary_condition_wall)

###############################################################################
# Get the DG approximation space

volume_flux = (flux_central, flux_nonconservative_gaburro)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_gaburro),
                 volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

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
