
export prodlaws, prodprocess

"""
Generate all permutations between discrete probabilities specified in args.

# Usage

```julia
julia> prodlaws(law1, law2, ..., law_n)

```
# Arguments
* `law::NoiseLaw`:
    First law to consider
* `laws::Tuple(NoiseLaw)`:
    Other noiselaws

# Return
`output::NoiseLaw`

# Exemple
    `noiselaw_product(law1, law2)`
with law1 : P(X=x_i) = pi1_i
and  law1 : P(X=y_i) = pi2_i
return the following discrete law:
    output: P(X = (x_i, y_i)) = pi1_i * pi2_i

"""
function prodlaws(law::DiscreteLaw, laws...)
    if length(laws) == 1
        # Read first law stored in tuple laws:
        n2 = laws[1]
        # Get support size of these two laws:
        nw1 = length(law)
        nw2 = length(n2)
        # and dimensions of aleas:
        n_dim1 = ndims(law)
        n_dim2 = ndims(n2)

        # proba and support will defined the output discrete law
        proba = zeros(nw1*nw2)
        support = zeros(nw1*nw2, n_dim1 + n_dim2)

        count = 1
        # Use an iterator to find all permutations:
        for tup in Base.product(1:nw1, 1:nw2)
            i, j = tup
            # P(X = (x_i, y_i)) = pi1_i * pi2_i
            proba[count] = law.probas[i] * n2.probas[j]
            support[count, :] = vcat(law.support[i, :], n2.support[j, :])
            count += 1
        end
        return DiscreteLaw(support, proba)
    else
        # Otherwise, compute result with recursivity:
        return prodlaws(law, prodlaws(laws[1], laws[2:end]...))
    end
end


"""Product of two whitenoise process.

# Notes
Process are supposed to be independent.

"""
function prodprocess(w1::WhiteNoise, w2::WhiteNoise)
    # chk consistency
    @assert length(w1) == length(w2)
    return WhiteNoise(DiscreteLaw[prodlaws(w1[t], w2[t]) for t in 1:length(w1)])
end
function prodprocess(wn::Vector{WhiteNoise})
    if length(wn) == 1
        return wn[1]
    elseif length(wn) == 2
        return prodprocess(wn[1], wn[2])
    else
        return prodprocess(wn[1], prodprocess(wn[2:end]))
    end
end

"Reduction of vector of noise processes.

When prodprocess between the noise processes is computationaly impossible, we apply a recursive three-by-three reduction by applying a k-means algorithm."
function reducelaws(nodesnoises::Array; nbins::Int=10)
    nnodes = length(nodesnoises)
    
    if nnodes == 1
        return nodesnoises[1]
    elseif nnodes <= 3
        # get total number of uncertainties
        nw = sum([size(noises.laws[1].support,2) for noises in nodesnoises])
        
        # get number of times steps
        ntimes = length(nodesnoises[1].laws)

        globallaw = Scenarios.prodprocess([noises for noises in nodesnoises])
        nscen = length(globallaw.laws[1])
        scenarios = zeros(Float64, ntimes, nscen, nw)

        for t in 1:ntimes
            scenarios[t, :, :] = globallaw.laws[t].support
        end

        weights = zeros(Float64, ntimes, nscen)
        for t in 1:ntimes
            weights[t,:] = globallaw.laws[t].probas.values
        end

        # then, we quantize vectors in `nw` dimensions
        return WhiteNoise(scenarios, weights=weights, nbins, KMeans())
    else
        #We split the nodes in three almost-equal parts
        newsize = Int(floor(nnodes/3))
        
        nodesnoises1 = nodesnoises[1:newsize]
        nodesnoises2 = nodesnoises[(newsize+1):(2*newsize)]
        nodesnoises3 = nodesnoises[(2*newsize+1):end]

        globallaw1 = reducelaws(nodesnoises1, nbins=10)
        globallaw2 = reducelaws(nodesnoises2, nbins=10)
        globallaw3 = reducelaws(nodesnoises3, nbins=10)

        newnodesnoises = vcat([globallaw1, globallaw2, globallaw3]...)

        return reducelaws(newnodesnoises, nbins=nbins)

    end
end