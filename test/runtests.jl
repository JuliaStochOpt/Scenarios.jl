

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

    # test weights
    p = Scenarios.weights(μ)
    @test sum(p) ≈ 1.

    # test resampling
    μ2 = resample(μ, 2)
    @test isa(μ2, DiscreteLaw)
    @test length(μ2) == 2
end


@testset "White noise" begin
    ntime = 10
    # generate random scenarios
    scen = rand(ntime, 100, 1)

    w = Scenarios.WhiteNoise(scen, 2, KMeans())

    @test isa(w, Scenarios.WhiteNoise)
    @test length(w) == ntime
    @test size(rand(w, 2)) == (10, 2, 1)


    # test product with 2 whitenoise
    w2 = Scenarios.WhiteNoise(scen, 2, KMeans())
    wprod = prodprocess(w, w2)
    @test isa(wprod, WhiteNoise)
    @test length(wprod) == length(w)

    # test product with 3 whitenoise
    w2 = Scenarios.WhiteNoise(scen, 2, KMeans())
    w3 = Scenarios.WhiteNoise(scen, 2, KMeans())
    wprod = prodprocess([w, w2, w3])

    # test resampling
    ws = Scenarios.WhiteNoise(scen, 5, KMeans())
    wnew = resample(ws, 2)
    @test isa(wnew, WhiteNoise)
    @test any(length.(wnew.laws) .== 2)

    # test resampling of vector of processes
    wnew = resample([ws, ws], 2)
    @test isa(wnew, WhiteNoise)
    @test any(length.(wnew.laws) .== 2)
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

    # Test that transition matrix is Markovian
    for t in 2:ntime-1
        @test sum(sum(m.transition[t, :, :], 2)) == 5
    end
end


@testset "Resampler" begin
    ntime = 10
    nprocesses = 6
    # generate random scenarios
    scen = rand(ntime, 100, 1)

    μtot = Scenarios.WhiteNoise[]
    for i = 1:nprocesses
        w = Scenarios.WhiteNoise(scen, 2, KMeans())
        push!(μtot, w)
    end

    # total size is 2^6 = 64
    # if quantization size is bigger than 64, an error is thrown
    rs = DiscreteLawSampler(65, 2)
    @test_throws ErrorException rs(μtot)

    # single bin quantization
    rs = DiscreteLawSampler(1, 2)
    @test isa(rs(μtot), Scenarios.WhiteNoise)

    rs = DiscreteLawSampler(10, 2)
    @test isa(rs(μtot), Scenarios.WhiteNoise)
end
