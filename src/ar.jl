

"""
    fitar(scenarios::Array{T, 2}, order=1)

Fit non-stationary AR process onto `scenarios`.
"""
function fitar{T}(scenarios::Array{T, 2}; order::Int=1)
    ntime = size(scenarios, 1)
    @assert order < ntime

    # Define non-stationary AR model such that :
    #    X[t+1] = α[t] * X[t] + β[t] + σ[t] * N(0, 1)
    α = zeros(T, ntime, order)
    β = zeros(T, ntime)
    σ = zeros(T, ntime)

    for t in (order):ntime-1
        sol = llsq(scenarios[t-order+1:t, :]', scenarios[t+1, :])
        α[t, :] = sol[1:order]
        β[t] = sol[end]

        ypred =   scenarios[t-order+1:t, :]' * α[t, :]  .+ β[t]
        σ[t] = std(ypred - scenarios[t+1, :])
    end

    return α, β, σ
end
