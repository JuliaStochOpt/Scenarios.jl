# Define Markov Chain

export MarkovChain

struct MarkovChain{S, T} <: AbstractStochasticProcess
    seed::StatsBase.ProbabilityWeights
    transition::Array{S, 3}
    support::Array{T, 3}
end


function MarkovChain{T}(scenarios::Array{T, 3}, nbins::Int, algo::AbstractQuantizer)
    ntime, nscenarios, nnoise = size(scenarios)

    transition = zeros(ntime-1, nbins, nbins)
    chainvalues = zeros(ntime, nbins, nnoise)
    # local transition matrix
    πij = zeros(Float64, nbins, nbins)
    # local counter
    ni = zeros(Int, nbins)

    probaold, supportold, flagsold = quantize(algo, scenarios[1, :, :]', nbins)
    chainvalues[1, :, :] .= supportold
    seed = probaold

    for t in 2:ntime
        proba, support, flags = quantize(algo, scenarios[t, :, :]', nbins)

        @inbounds for s in 1:nscenarios
            πij[flagsold[s], flags[s]] += 1
            ni[flagsold[s]] += 1
        end

        # rescale proba
        πij = πij ./ ni

        # store results in matrix
        transition[t-1, :, :] .= πij
        chainvalues[t, :, :] .= supportold

        # update previous values to current
        supportold .= support
        flagsold .= flags
    end

    MarkovChain(pweights(seed), transition, chainvalues)
end


Base.length(m::MarkovChain) = size(m.transition, 1)
Base.size(m::MarkovChain) = size(m.support)


function Base.rand{S, T}(m::MarkovChain{S, T})
    val = zeros(T, size(m)[2:3]...)

    # initiate first value
    index = sample(m.seed)
    val[1, :] = m.support[1, index, :]

    for t = 2:length(m)+1
        index = sample(pweights(m.transition[t-1, index, :]))
        val[t, :] = m.support[t, index, :]
    end

    val
end
