
module Scenarios

using Clustering, JuMP, StatsBase, MultivariateStats, NearestNeighbors

include("quantizers.jl")
include("types.jl")
include("distance.jl")
include("ar.jl")
include("markov.jl")

end
