# a search algorithm that creates a diverse set of boundary candidates.
# 1. import training dataset (binary, tableau, continuous inputs, 2d)
# 2. randomly sample input pairs from both categories (1/0).
# 3. use global search to "squeeze out" a boundary (requires definition of a "satisfactory" delta in the output).
#   - use Program Derivative for the squeeze in the fitness function, using standard EA search (diffevo).

using BlackBoxOptim

# Get upper and lower bounds for searching among the input columns
function get_bounds(df, InputColumns)
    T = typeof(df[1, InputColumns[1]])
    lbs = zeros(T, length(InputColumns))
    hbs = zeros(T, length(InputColumns))
    for (i, icol) in enumerate(InputColumns)
        lbs[i] = minimum(df[:, icol])
        hbs[i] = maximum(df[:, icol])
    end
    return lbs, hbs
end

# As input distance we just use Euclidean, for now. This is probably not good
# in the general case but we just want to get going.
using Distances
const EuclideanDist = Euclidean()

function convergence_callback(c::BlackBoxOptim.OptRunController)
    bc = best_candidate(c)
    if best_fitness(c) < Inf # valid pair (boundary has candidate on two sides of the boundary)
        dist = EuclideanDist((bc[1], bc[2]), (bc[3], bc[4])) # FIXME dist metric currently set on two places
        
        if dist < Delta
                c.max_steps = BlackBoxOptim.num_steps(c)-1
                println("--------")
                println(c.max_steps)
                println(best_fitness(c))
                println(bc)
                println("--------")
        end
    end
end

using StatsBase # sample
# TODO in the binary example it is always the same, but for multiple classes, there may be a variety of ways to implement it
function draw_init_population(ClassPoints::Dict, N=20)

    classes = map(_ -> sample(collect(keys(ClassPoints)) , 2, replace=false), 1:N)
    classes = sort.(classes) # TODO for 3++ classes, this can result in any kind of combinations on the positions.

    points = map(p -> [rand(Points[p[1]]), rand(Points[p[2]])], classes)

    result = Matrix{Float64}(undef, 4, N)
    
    for (i, r) in enumerate(points)
            result[1,i] = r[1][1]
            result[2,i] = r[1][2]
            result[3,i] = r[2][1]
            result[4,i] = r[2][2]
    end
    
    return result
end

function diverse_bcs(fitness::Function, ClassPoints::Dict, iterations::Int=500, initial_candidates::Int=20, searchrange = SearchRangePair)
    cands = []
    removenext = 1
    doremovenext = true
    incumbent_diversity_diff = 0
    for i in 1:iterations
        global InitPopulation = draw_init_population(ClassPoints) # FIXME generalize
        res = bboptimize(fitness; SearchRange = searchrange,
                                    CallbackInterval = 0.0,
                                    CallbackFunction = convergence_callback,  # FIXME generalize
                                    MaxTime = .3,
                                    Population = InitPopulation)
        cand = best_candidate(res)
        #TODO have a check for whether the search was successful... if points too far apart, not successf. also interesting - what are those cases? -> investigate.

        if length(cands) < initial_candidates + 1
            push!(cands, cand)
        else
            if !doremovenext
                push!(cands, cands[removenext])
                doremovenext = true
            end

            cands[removenext] = cand
            apoints = map(c -> c[1:2], cands)
            distances = pairwise(EuclideanDist, apoints)
            neighbordistancesums = sum(sort(distances, dims=2)[:,2:3], dims=2)[:,1]

            diversity_diff = sum(neighbordistancesums) / length(cands)

            if incumbent_diversity_diff < diversity_diff
                incumbent_diversity_diff = diversity_diff
                doremovenext = false
            end

            removenext = argmin(neighbordistancesums) # remove closest to its two neighbor points (2d boundary has 2 neighbor points)
        end
        "************ $i" |> println
    end

    if doremovenext
        cands = cands[setdiff(1:end, removenext)] # ensure that the very last "removenext" is respected
    end

    return cands
end

isdifferent(a,b) = a == b ? 0 : 1
const Delta = 1e-6
const FitnessBreakpoint = 10.0
function pd(pointA, pointB, classA, classB;
    dist_output = isdifferent, #TODO allow for ordinal distance or any other measure.
    dist_input = EuclideanDist)

    dout = dist_output(classA, classB)
    din = dist_input(pointA, pointB)

    if dout == 0.0
        return Inf # maximially penalize if same category - not our intention.
        #return FitnessBreakpoint + 1.0 / (din + Delta) # Push points away until we hopefully find a difference in the outputs
    else
        return FitnessBreakpoint - dout / (din + Delta) # When we have some output distance we start giving a benefit to points being close.
    end
end

#TODO how to do if more than two classes!? conceptualize to understand difference?
function pd_fitness(validity_check::Function)
    return (x::Vector{Float64}) -> begin
        classA = validity_check(x[1], x[2])
        classB = validity_check(x[3], x[4])
        pd((x[1], x[2]), (x[3], x[4]), classA, classB)
    end
end

function config_search(DF, classcol::Symbol=:class)
    lowerbounds, higherbounds = get_bounds(DF, Inps)
    SearchRangePoint = collect(zip(lowerbounds, higherbounds))
    SearchRangePair = vcat(SearchRangePoint, SearchRangePoint)
    
    classes = unique(DF[!, classcol])

    Points = Dict()
    for cl in classes
        Points[cl] = map(r -> (r.x1, r.x2), eachrow(filter(r -> r[classcol] == cl, DF)))
    end

    return SearchRangePair, Points
end