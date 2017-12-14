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

    probaold, supportold, flagsold = quantize(algo, scenarios[1, :, :]', nbins)
    chainvalues[1, :, :] .= supportold
    seed = probaold

    for t in 2:ntime
        proba, support, flags = quantize(algo, scenarios[t, :, :]', nbins)

        πij = 0 .* πij

        @inbounds for s in 1:nscenarios
            πij[flagsold[s], flags[s]] += 1
        end

        # rescale proba
        πij = πij ./ sum(πij, 2)

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
    val = zeros(T, size(m)[1], size(m)[3])

    # initiate first value
    index = sample(m.seed)
    val[1, :] = m.support[1, index, :]

    for t = 2:length(m)+1
        index = sample(pweights(m.transition[t-1, index, :]))
        val[t, :] = m.support[t, index, :]
    end

    val
end

function Base.rand{S, T}(m::MarkovChain{S, T}, n_s::Int)

    vals = zeros(T, size(m)[1], n_s, size(m)[3])
    for s in 1:n_s
        vals[:,s,:] = rand(m)
    end
    vals
end

function Base.rand{S, T}(m::MarkovChain{S, T}, t_0::Int, x_0::Vector{T})
    val = zeros(T, size(m)[1]-t_0+1, size(m)[3])

    # initiate first value
    kdtree = KDTree(m.support[1,:,:]')

    index = knn(kdtree, x_0, 1)[1][1]
    
    val[1, :] = m.support[1, index, :]

    for t = 2:length(val)
        index = sample(pweights(m.transition[t-1, index, :]))
        val[t, :] = m.support[t, index, :]
    end

    val
end

function Base.rand{S, T}(m::MarkovChain{S, T}, n_s::Int, t_0::Int, x_0::Vector{T})
    vals = zeros(T, size(m)[1]-t_0+1, n_s, size(m)[3])
    for s in 1:n_s
        vals[:,s,:] = rand(m, t_0, x_0)
    end
    vals
end