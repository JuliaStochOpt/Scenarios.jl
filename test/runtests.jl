

using Scenarios

using Base.Test


@testset "Probability law" begin

    sizelaw = 10
    sup = rand(sizelaw)
    prob = rand(sizelaw)
    μ = DiscreteLaw(sup, prob)
    @test isa(μ, DiscreteLaw)
    @test ndims(μ) == 1
    @test length(μ) == 10

    μ = DiscreteLaw(sup)
    @test length(μ) == 1

    sup = rand(sizelaw, 2)
    prob = rand(sizelaw)
    μ = DiscreteLaw(sup, prob)
    @test isa(μ, DiscreteLaw)
    @test ndims(μ) == 2
    @test length(μ) == 10

    μprod = prodlaws(μ, μ)
    @test isa(μprod, DiscreteLaw)
    @test ndims(μprod) == 4
    @test length(μprod) == 100
end


@testset "White noise" begin

    ntime = 10
    # generate random scenarios
    scen = rand(ntime, 100, 1)

    w = Scenarios.WhiteNoise(scen, 2, KMeans())

    @test isa(w, Scenarios.WhiteNoise)
    @test length(w) == ntime
    @test size(rand(w, 2)) == (10, 2, 1)
end


@testset "AR process" begin
    ntime = 10
    scen = rand(ntime, 100)

    a, b, s = fitar(scen, order=1)
    @test size(a) == (ntime-1, 1)

    a, b, s = fitar(scen, order=2)
    @test size(a) == (ntime-1, 2)
end


@testset "Markov Chain" begin
    ntime = 10
    nbins = 5
    scen = rand(ntime, 100, 2)

    m = MarkovChain(scen, nbins, KMeans())

    simu = rand(m)
    @test size(simu) == (ntime, 2)

end
