

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
end
