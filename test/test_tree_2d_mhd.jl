module TestExamples2DMHD

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "MHD" begin
#! format: noindent

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            0.00011149543672225127,
                            5.888242524520296e-6,
                            5.888242524510072e-6,
                            8.476931432519067e-6,
                            1.3160738644036652e-6,
                            1.2542675002588144e-6,
                            1.2542675002747718e-6,
                            1.8705223407238346e-6,
                            4.651717010670585e-7,
                        ],
                        linf=[
                            0.00026806333988971254,
                            1.6278838272418272e-5,
                            1.627883827305665e-5,
                            2.7551183488072617e-5,
                            5.457878055614707e-6,
                            8.130129322880819e-6,
                            8.130129322769797e-6,
                            1.2406302192291552e-5,
                            2.373765544951732e-6,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            1.7201098719531215e-6,
                            8.692057393373005e-7,
                            8.69205739320643e-7,
                            1.2726508184718958e-6,
                            1.040607127595208e-6,
                            1.07029565814218e-6,
                            1.0702956581404748e-6,
                            1.3291748105236525e-6,
                            4.6172239295786824e-7,
                        ],
                        linf=[
                            9.865325754310206e-6,
                            7.352074675170961e-6,
                            7.352074674185638e-6,
                            1.0675656902672803e-5,
                            5.112498347226158e-6,
                            7.789533065905019e-6,
                            7.789533065905019e-6,
                            1.0933531593274037e-5,
                            2.340244047768378e-6,
                        ],
                        volume_flux=(flux_derigs_etal, flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
                        l2=[
                            3.7762324533854616e-6,
                            1.5534623833573546e-6,
                            1.4577234868196855e-6,
                            1.7647724628707057e-6,
                            1.4831911814574333e-6,
                            1.456369119716533e-6,
                            1.4115666913995062e-6,
                            1.804758237422838e-6,
                            8.320469738087189e-7,
                        ],
                        linf=[
                            3.670661330201774e-5,
                            1.530289442645827e-5,
                            1.3592183785327006e-5,
                            1.5173897443654383e-5,
                            9.43771379136038e-6,
                            1.0906323046233624e-5,
                            1.0603954940346938e-5,
                            1.5900499596113726e-5,
                            5.978772247650426e-6,
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[
                            0.03637302248881514,
                            0.043002991956758996,
                            0.042987505670836056,
                            0.02574718055258975,
                            0.1621856170457943,
                            0.01745369341302589,
                            0.017454552320664566,
                            0.026873190440613117,
                            5.336243933079389e-16,
                        ],
                        linf=[
                            0.23623816236321427,
                            0.3137152204179957,
                            0.30378397831730597,
                            0.21500228807094865,
                            0.9042495730546518,
                            0.09398098096581875,
                            0.09470282020962917,
                            0.15277253978297378,
                            4.307694418935709e-15,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_orszag_tang.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
                        l2=[
                            0.21967600768935716,
                            0.2643126515795721,
                            0.31488287201980875,
                            0.0,
                            0.5160141621186931,
                            0.23028914748088603,
                            0.34413527376463915,
                            0.0,
                            0.003178793090381426,
                        ],
                        linf=[
                            1.2749969218080568,
                            0.6737013368774057,
                            0.8604154399895696,
                            0.0,
                            2.799342099887639,
                            0.6473347557712643,
                            0.9691773375490476,
                            0.0,
                            0.05729832038724348,
                        ],
                        tspan=(0.0, 0.09))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_orszag_tang.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
                        l2=[
                            0.10806619664693064,
                            0.20199136742199922,
                            0.22984589847526207,
                            0.0,
                            0.29950152196422647,
                            0.15688413207147794,
                            0.24293641543490646,
                            0.0,
                            0.003246181006326598,
                        ],
                        linf=[
                            0.560316034595759,
                            0.5095520363866776,
                            0.6536748458764621,
                            0.0,
                            0.9627447086204038,
                            0.3981375420906146,
                            0.673472146198816,
                            0.0,
                            0.04879208429337193,
                        ],
                        tspan=(0.0, 0.06),
                        surface_flux=(flux_hll, flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl one step with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            7.144325530681224e-17,
                            2.123397983547417e-16,
                            5.061138912500049e-16,
                            3.6588423152083e-17,
                            8.449816179702522e-15,
                            3.9171737639099993e-16,
                            2.445565690318772e-16,
                            3.6588423152083e-17,
                            9.971153407737885e-17,
                        ],
                        linf=[
                            2.220446049250313e-16,
                            8.465450562766819e-16,
                            1.8318679906315083e-15,
                            1.1102230246251565e-16,
                            1.4210854715202004e-14,
                            8.881784197001252e-16,
                            4.440892098500626e-16,
                            1.1102230246251565e-16,
                            4.779017148551244e-16,
                        ],
                        maxiters=1,
                        initial_condition=initial_condition_constant,
                        atol=2.0e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                        l2=[
                            1.2623319195262743,
                            1.8273050553090515,
                            1.7004151198284634,
                            0.0,
                            2.2978570581460818,
                            0.2147235065899803,
                            0.23558337696054493,
                            0.0,
                            0.0032515115395693483,
                        ],
                        linf=[
                            11.003677581472843,
                            14.70614192714736,
                            15.687648666952708,
                            0.0,
                            17.098104835553823,
                            1.3283750501377847,
                            1.4365828094434892,
                            0.0,
                            0.07886241196068537,
                        ],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
                        l2=[
                            0.17646728395490927,
                            3.866230215339417,
                            2.4867304651291255,
                            0.0,
                            355.4562971958441,
                            2.359493623565687,
                            1.4030741420730297,
                            0.0,
                            0.029613599942667133,
                        ],
                        linf=[
                            1.581630420824181,
                            44.15725488910748,
                            13.056964982196554,
                            0.0,
                            2244.875490238186,
                            13.07679044647926,
                            9.14612176426092,
                            0.0,
                            0.5154756722488522,
                        ],
                        tspan=(0.0, 0.003),
                        # Calling the AnalysisCallback before iteration 9 causes the interpolation
                        # of this IC to have negative density/pressure values, crashing the simulation.
                        coverage_override=(maxiters = 9,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

end # module
