
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

function calc_error_h(deg, level)
    ###############################################################################
    # Konvergenzanalyse mit der manufactured solutions Methode, Solution H(x,t) = sin(2*pi*(x-t))+5, 
    # source term ergibt sich durch Einsetzen in die Flachwassergleichung. Term g*h*b_x wird mit eingerechnet

    equations = ShallowWaterEquations1D(gravity_constant=1.0)

    function initial_condition_manufactured(x, t, equations)

        H = sin(2 * pi * (x[1] - t)) + 5.0
        v = 1.0
        b = sin(2 * pi * x[1]) + 2.0
    
        return prim2cons(SVector(H, v, b), equations)
    end
    
    function source_terms_manufactured(u, x, t, equations)
    
        H = sin(2 * pi * (x[1] - t)) + 5.0
        b = sin(2 * pi * x[1]) + 2.0
    
        b_x = 2 * pi * cos(2* pi * x[1])
        H_x =  2 * pi * cos(2 * pi * (x[1] - t))
        H_t = -2 * pi * cos(2 * pi * (x[1] - t))
    
        du1 = H_t + H_x - b_x
        du2 = du1 + (H - b) * H_x
        return SVector(du1, du2, 0.0)
    
    end    

    ###############################################################################

    # Get the DG approximation space


    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)


    solver = DGSEM(polydeg=deg, surface_flux=(flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))


    ###############################################################################

    # Get the TreeMesh and setup a periodic mesh


    coordinates_min = 0.0

    coordinates_max = 1.0

    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level,
                n_cells_max=10_000)

        
    # Create the semi discretization object

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_manufactured, solver, source_terms=source_terms_manufactured)

    ###############################################################################
    # ODE solver

    tspan = (0.0, 1.0)
    ode = semidiscretize(semi, tspan)

    # Callbacks
    stepsize_callback = StepsizeCallback(cfl=0.5)

    callbacks = CallbackSet(stepsize_callback)

    ###############################################################################
    # run the simulation

    sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, abstol=1.0e-10, reltol=1.0e-10, save_everystep=false, callback=callbacks);

    error_all = abs.(sol[1] - sol[2])

    error_h = zeros((deg + 1) * 2^level)

    for i = 1:((deg + 1) * 2^level)
        error_h[i] = error_all[(i-1)*3 + 1]
    end

    return norm(error_h, Inf)

end


function konvergenz_tabelle(deg)
    
    polynomdeg = deg
    N_Q = [1,2,3,4,5,6,7,8]
    if deg == 7
        N_Q = [1,2,3,4,5,6]
    end
    error = zeros(length(N_Q))
    eoc = zeros(length(N_Q))

    global j = 1
    for level in N_Q
    
        error[j] = calc_error_h(polynomdeg, level)
    
        if j > 1
            eoc[j] = log(error[j]/error[j-1])/log(2^(j-2)/2^(j-1))
        end
        global j += 1
    end

    println("Polynomgrad =  ", polynomdeg)
    println("------------------------------------------------------------------")
    println("|   N_Q   |            Error           |           EOC           |")
    println("------------------------------------------------------------------")

    for i = 1:length(N_Q)
        println(2^N_Q[i],"  ", error[i],"  ", eoc[i])
    end

end