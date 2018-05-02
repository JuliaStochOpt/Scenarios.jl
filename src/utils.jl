
export prodlaws, prodprocess
export resample, DiscreteLawSampler

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
    # chk consistency: process must share same number of timesteps
    @assert length(w1) == length(w2)
    return WhiteNoise(DiscreteLaw[prodlaws(w1[t], w2[t]) for t in 1:length(w1)])
end
function prodprocess(wn::Vector{WhiteNoise})
    # get maximum support size of product
    maxsize = prod([maximum(length.(w.laws)) for w in wn])
    (maxsize > 100_000) && error("Final support size is too large: greater than 100_000")

    if length(wn) == 1
        return wn[1]
    elseif length(wn) == 2
        return prodprocess(wn[1], wn[2])
    else
        return prodprocess(wn[1], prodprocess(wn[2:end]))
    end
end

# raw resampling
function resample(mu::DiscreteLaw, nbins::Int)
    # first, test if resampling is valid
    ~(1 <= nbins <= length(mu)) && error("Wrong resampling: we should have 1 <= nbins <= $(length(mu))")
    # then, disjunction wrt the different case
    if nbins == 1
        proba = weights(mu)
        sup = zeros(Float64, 1, ndims(mu))
        sup[:] = (mu.support' * proba)
        return DiscreteLaw(sup, [1.])
    elseif nbins == length(mu)
        return mu
    else
        k = kmeans(mu.support', nbins, weights=weights(mu))
        return DiscreteLaw(k.centers', k.cweights)
    end
end

resample(w::WhiteNoise, nbins::Int) = WhiteNoise(resample.(w.laws, nbins))
resample(w::Vector{WhiteNoise}, nbins::Int) = resample(prodprocess(w), nbins)

# resampler
abstract type AbstractSampler end

struct DiscreteLawSampler <: AbstractSampler
    # final quantization size
    nbins::Int
    # intermediate quantization size (for subprocess)
    nbins_inner::Int
    # subprocesses length
    Δn::Int
    # maximum allowable size for intermediate sampling
    MAXSIZE::Int
end
DiscreteLawSampler(nbins, nbins_inner; Δn=-1) = DiscreteLawSampler(nbins, nbins_inner, Δn, 100_000)

function (sampler::DiscreteLawSampler)(noises::Vector{WhiteNoise})
    # if not specified, resample all in once
    if (sampler.Δn == -1 ) || (sampler.Δn >= length(noises))
        return resample(noises, sampler.nbins)
    end

    # otherwise, resample iteratively the noise process
    nwindows = ceil(Int, length(noises) / sampler.Δn)
    Δn = sampler.Δn

    # ensure that total prodprocess size is not too big
    @assert sampler.nbins_inner^nwindows <= sampler.MAXSIZE

    # first step
    ## we resample the noise sequentially
    i0, i1 = 1, Δn

    μtot = WhiteNoise[]

    for nw in 1:nwindows
        μ = resample(noises[i0:i1], sampler.nbins_inner)
        push!(μtot, μ)
        i0 += Δn
        i1 = min(i1+Δn, length(noises))
    end

    # second step
    ## resample inside the resampled subprocess
    return resample(μtot, sampler.nbins)
end
