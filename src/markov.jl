# Define Markov Chain

export MarkovChain

struct MarkovChain{S, T} <: AbstractStochasticProcess
    seed::StatsBase.ProbabilityWeights
    transition::Array{S, 3}
    support::Array{T, 3}
end


function MarkovChain(scenarios::Array{T, 3}, nbins::Int, algo::AbstractQuantizer) where T
    ntime, nscenarios, nnoise = size(scenarios)

    transition = zeros(ntime-1, nbins, nbins)
    chainvalues = zeros(ntime, nbins, nnoise)
    # local transition matrix
    πij = zeros(Float64, nbins, nbins)

    probaold, supportold, flagsold = quantize(algo, collect(scenarios[1, :, :]'), nbins)
    chainvalues[1, 1:size(supportold)[1], :] .= supportold
    seed = probaold

    for t in 2:ntime
        proba, support, flags = quantize(algo, collect(scenarios[t, :, :]'), nbins)

        πij[:] = 0.

        @inbounds for s in 1:nscenarios
            πij[flagsold[s], flags[s]] += 1
        end

        # rescale proba
        for i in 1:nbins
            sum_transit = sum(πij[i, :])
            if sum_transit > 1e-10
                πij[i, :] = πij[i, :] / sum_transit
            end
        end

        # store results in matrix
        transition[t-1, :, :] .= πij
        chainvalues[t, 1:size(supportold)[1], :] .= supportold

        # update previous values to current
        supportold = support

        flagsold .= flags
    end

    MarkovChain(pweights(seed), transition, chainvalues)
end


Base.length(m::MarkovChain) = size(m.transition, 1)
Base.size(m::MarkovChain) = size(m.support)


function Base.rand(m::MarkovChain{S, T}) where {S, T}
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

function Base.rand(m::MarkovChain{S, T}, n_s::Int) where {S, T}

    vals = zeros(T, size(m)[1], n_s, size(m)[3])
    for s in 1:n_s
        vals[:,s,:] = rand(m)
    end
    vals
end

function Base.rand(m::MarkovChain{S, T}, t_0::Int, x_0::Vector{T}) where {S, T}
    val = zeros(T, size(m)[1]-t_0+1, size(m)[3])

    val[1, :] = x_0

    # find first closest centroid index
    kdtree = KDTree(collect(m.support[1,:,:]'))

    index = knn(kdtree, x_0, 1)[1][1]

    for t = 2:length(val)
        index = sample(pweights(m.transition[t-1, index, :]))
        val[t, :] = m.support[t, index, :]
    end

    val
end

function Base.rand(m::MarkovChain{S, T}, n_s::Int, t_0::Int, x_0::Vector{T}) where {S, T}
    vals = zeros(T, size(m)[1]-t_0+1, n_s, size(m)[3])
    for s in 1:n_s
        vals[:,s,:] = rand(m, t_0, x_0)
    end
    vals
end

function forecast(m::MarkovChain{S,T}, t_0::Int, x_0::Vector{T}) where {S, T}
    val = zeros(T, size(m)[1]-t_0+1, size(m)[3])

    val[1, :] = x_0

    # find first closest centroid index
    kdtree = KDTree(collect(m.support[1,:,:]'))

    index = knn(kdtree, x_0, 1)[1][1]

    for t = 2:length(val)

        for nw in 1:size(val)[2]
            val[t, nw] = dot(m.support[t, :, nw], m.transition[t-1, index, :])
        end

        kdtree = KDTree(collect(m.support[t,:,:]'))

        index = knn(kdtree, val[t,:], 1)[1][1]

    end

    val
end
