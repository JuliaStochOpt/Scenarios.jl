# Define quantizers

abstract type AbstractQuantizer end

immutable KMeans end

"""
    quantize(algo::AbstractQuantizer, points::Array{T, 2}, nbins::Int)

Quantize `points` in `nbins` with `algo` quantizer.
"""
function quantize end

function quantize(::KMeans, points, nbins)
    R = kmeans(points, nbins)
    valid = R.counts .> 1e-6
    return R.counts[valid] ./ sum(R.counts[valid]), R.center[:, valid]'
end
