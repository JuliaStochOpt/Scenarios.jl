
"""Compute Wasserstein distance between distribution `P` and `Q`."""
function WassersteinDistance(P::DiscreteLaw, Q::DiscreteLaw, solver)
    μ = P.probas
    ν = Q.probas
    n = length(μ)
    m = Model(solver = solver)
    @variable(m, p[1:n,1:n] >= 0)
    @objective(m, Min, sum(p[i,j] for i in 1:n for j in 1:n if i != j))
    for k in 1:n
        @constraint(m, sum(p[k,:]) == μ[k])
        @constraint(m, sum(p[:,k]) == ν[k])
    end
    solve(m)
    getobjectivevalue(m)
end
