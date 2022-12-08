using Trixi, OrdinaryDiffEq, Plots, LinearAlgebra, BenchmarkTools

# Advektions-Geschwindigkeit
advection_velocity = 2.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Bestimmt die Eigenwerte der Systemmatrix und plottet diese und die skalierten Eigenwerte
function stabilityArea(deg, level, scalar)
    
    # create DG solver with flux lax friedrichs and LGL basis
    solver = DGSEM(polydeg=deg, surface_flux=flux_lax_friedrichs)

    # Diskretisieren mit `TreeMesh`
    coordinates_min = 0 
    coordinates_max = 1.0 
    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level, # number of elements = 2^refinement_level
                n_cells_max=30_000)
                
    # create initial condition and semidiscretization
    initial_condition_sine_wave(x, t, equations) = SVector(sin(2.0 * pi * sum(x)))
    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

    A, b = linear_structure(semi)

    λ = eigvals(Matrix(A))

    scatter(real.(λ), imag.(λ), label="Eigenwerte von A")

    λ_scaled = scalar .* λ

    scatter!(real.(λ_scaled), imag.(λ_scaled), label="skalierte Eigenwerte von A")

    # andere Variante die Systemmatrix ausgeben zu lassen:
    #A = jacobian_ad_forward(semi);
    #λ = eigvals(A)

    stab_poly(x) = 1 + x + 1/2 * x*x + 1/6 *x*x*x + 1/24 * x*x*x*x + 1/200 *x*x*x*x*x

    eigvals_poly = abs.(stab_poly.(λ_scaled))
    max_val_poly = norm(eigvals_poly, Inf)

    println("Maximaler Wert des Betrags des Stabilitätspolynoms an den skalierten EW:", max_val_poly)

end


function error_calc(deg, level)
    
    # create DG solver with flux lax friedrichs and LGL basis
    solver = DGSEM(polydeg=deg, surface_flux=flux_lax_friedrichs)

    # Diskretisieren mit `TreeMesh`
    coordinates_min = 0 
    coordinates_max = 1.0 
    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level, # number of elements = 2^refinement_level
                n_cells_max=30_000)
                
    # create initial condition and semidiscretization
    function initial_condition_sine_wave(x, t, equations) 
        println(x)
        println(SVector(sin(2.0 * pi * sum(x))))
        return SVector(sin(2.0 * pi * sum(x))) 
    end
    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

    # solve
    tspan = (0.0, 1.0)
    ode_trixi  = semidiscretize(semi, tspan)
    sol_trixi  = solve(ode_trixi, RDPK3SpFSAL49(), abstol=1.0e-10, reltol=1.0e-10, save_everystep=false);

    error = abs.(vec(sol_trixi.u[1]) - vec(sol_trixi.u[2]))
    return norm(error,Inf)
end


function fehler_und_konvergenztabellen()
    error = zeros(3,8)
    eoc = zeros(3,8)
    polynomdeg = [1,3,7]
    N_Q = [0,1,2,3,4,5,6,7]
    global i = 1
    global j = 1
    for deg in polynomdeg
        global j = 1
        println("Polynomgrad: ", deg)
        for level in N_Q
            error[i,j] = error_calc(deg, level)
            if j > 1
                eoc[i,j] = log(error[i,j]/error[i,j-1])/log(2^(j-2)/2^(j-1))
            end
            println("  N_Q: ", level, "  Fehler: ", error[i,j])
            global j += 1
        end
        global i += 1
    end

    for k in 1:3
        println("Polynomgrad: ", polynomdeg[k])
        for l in 1:8
            println("   N_Q: ", N_Q[l],  "  Konvergenzordnung:  ", eoc[k,l])
        end
    end
end

function linear_start(deg,level)
    # create DG solver with flux lax friedrichs and LGL basis
    solver = DGSEM(polydeg=deg, surface_flux=flux_lax_friedrichs)

    # Diskretisieren mit `TreeMesh`
    coordinates_min = 0 
    coordinates_max = 1.0 
    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level, # number of elements = 2^refinement_level
                n_cells_max=30_000)
                
    # create initial condition and semidiscretization
    function initial_condition_sine_wave(x, t, equations)
        if 0.3 <= x[1] <= 0.7
            scalar = 1.0
        else 
            scalar = 0.0
        end
        return SVector(scalar)
    end
    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

    # solve
    tspan = (0.0, 2.0)
    ode_trixi  = semidiscretize(semi, tspan)
    sol_trixi  = solve(ode_trixi, RDPK3SpFSAL49(), abstol=1.0e-10, reltol=1.0e-10, save_everystep=false);

    return sol_trixi;
end


function plot_sin(deg, level)
    # create DG solver with flux lax friedrichs and LGL basis
    solver = DGSEM(polydeg=deg, surface_flux=flux_lax_friedrichs)

    # Diskretisieren mit `TreeMesh`
    coordinates_min = 0 
    coordinates_max = 1.0 
    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=level, # number of elements = 2^refinement_level
                n_cells_max=30_000)
                
    # create initial condition and semidiscretization
    initial_condition_sine_wave(x, t, equations) = SVector(sin(2.0 * pi * sum(x)))
    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

    # solve
    tspan = (0.0, 10.0)
    ode_trixi  = semidiscretize(semi, tspan)
    sol_trixi  = solve(ode_trixi, RDPK3SpFSAL49(), abstol=1.0e-10, reltol=1.0e-10, save_everystep=false);

    plot(sol_trixi, label="solution at t=$(tspan[2]) with Trixi.jl", legend=:topleft, linestyle=:dash, lw=2)

end
