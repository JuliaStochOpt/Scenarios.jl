# Define generic types to define noise

export DiscreteLaw, WhiteNoise

abstract type AbstractProbabilityLaw end

################################################################################
### Define discrete probability utility
struct DiscreteLaw{T} <: AbstractProbabilityLaw
    support::Array{T, 2}
    probas::StatsBase.ProbabilityWeights
end

function DiscreteLaw(support::Array{T, 2}, proba::Vector{T}) where T
    @assert size(support, 1) == length(proba)
    return DiscreteLaw(support, pweights(proba))
end
function DiscreteLaw(support::Vector{T}, proba::Vector{T}) where T
    support = reshape(support, length(support), 1)
    @assert size(support, 1) == length(proba)
    return DiscreteLaw(support, pweights(proba))
end
function DiscreteLaw(support::Vector{T}) where T
    support = reshape(support, 1, length(support))
    return DiscreteLaw(support, pweights([1.]))
end

Base.rand(law::DiscreteLaw) = law.support[StatsBase.sample(law.probas), :]
function Base.rand(laws::Vector{DiscreteLaw}, n::Int)
    scenarios = zeros(length(laws), n, ndims(laws[1]))
    for (t,law) in enumerate(laws)
        for i in 1:n
            scenarios[t, i, :] = rand(law)
        end
    end
    scenarios
end

Base.ndims(law::DiscreteLaw) = length(law.support[1, :])
Base.length(law::DiscreteLaw) = length(law.probas)

weights(law::DiscreteLaw) = law.probas.values ./ law.probas.sum




################################################################################
### Define stochastic processes utility
abstract type AbstractStochasticProcess end

struct WhiteNoise <: AbstractStochasticProcess
    laws::Vector{DiscreteLaw}
end

"""Quantize scenarios time step by time step."""
function WhiteNoise(scenarios::Array{Float64, 3}, nbins::Int, algo::AbstractQuantizer)
    T, n, Nw = size(scenarios)
    laws = DiscreteLaw[]

    for t in 1:T
        proba, support = quantize(algo, collect(scenarios[t, :, :]'), nbins)
        push!(laws, DiscreteLaw(support, proba))
    end

    WhiteNoise(laws)
end

Base.rand(noise::WhiteNoise,n::Int=1) = Base.rand(noise.laws, n)
Base.getindex(W::WhiteNoise, t::Int) = W.laws[t]
Base.length(W::WhiteNoise) = length(W.laws)
