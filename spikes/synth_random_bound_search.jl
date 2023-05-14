using CSV, DataFrames
using BlackBoxOptim

# a search algorithm that creates a diverse set of boundary candidates.
# 1. import training dataset (binary, tableau, continuous inputs, 2d)
# 2. randomly sample input pairs from both categories (1/0).
# 3. use global search to "squeeze out" a boundary (requires definition of a "satisfactory" delta in the output).
#   - use Program Derivative for the squeeze in the fitness function, using standard EA search (diffevo).

DataDir = joinpath(@__DIR__(), "..", "data")
const DF = CSV.read(joinpath(DataDir, "synth_bound.csv"), DataFrame)

# Inputs... then output
O = "class"
oidx = findfirst(==(O), names(DF))
Inps = names(DF)[1:(oidx-1)]

const Inputs = Matrix(DF[:, Inps])
const Outputs = DF[:, O]

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

lowerbounds, higherbounds = get_bounds(DF, Inps)
SearchRangePoint = collect(zip(lowerbounds, higherbounds))
const SearchRangePair = vcat(SearchRangePoint, SearchRangePoint)

# As input distance we just use Euclidean, for now. This is probably not good
# in the general case but we just want to get going.
using Distances
const EuclideanDist = Euclidean()

Class0Points = map(r -> (r.x1, r.x2), eachrow(filter(r -> r.class == 0, DF)))
Class1Points = map(r -> (r.x1, r.x2), eachrow(filter(r -> r.class == 1, DF)))
a, b = rand(Class0Points), rand(Class1Points)

isdifferent(a,b) = a == b ? 0 : 1
const Delta = 1e-6
const FitnessBreakpoint = 10.0
function pd(pointA, pointB, classA, classB;
    dist_output = isdifferent,
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

# fitness function: the format is floats... four floats, class is a consequence only.
# synthetic boundary described as bounding box built out of non-linear functions
synth_bound_func_left(x) = 5 + ((x+12)+2)*((x+12)^2-4)
synth_bound_func_right(x) = -5 + ((x-12)+2)*((x-12)^2-4)
synth_bound_func_up(x) = 10
synth_bound_func_down(x) = -17

function check_synth_bound_valid(x::Number,y::Number)
    if y > synth_bound_func_up(x) ||
        y < synth_bound_func_down(x) ||
        y > synth_bound_func_left(x) ||
        y < synth_bound_func_right(x)
        return false
    end
    return true
end

function fitness_function(x::Vector{Float64})
    classA = check_synth_bound_valid(x[1], x[2])
    classB = check_synth_bound_valid(x[3], x[4])
    pd((x[1], x[2]), (x[3], x[4]), classA, classB)
end

function mycallback(c::BlackBoxOptim.OptRunController)
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

function draw_init_population(Class0Points, Class1Points, N=20)
    as = collect(Iterators.flatten(rand(Class0Points, N)))
    bs = collect(Iterators.flatten(rand(Class1Points, N)))

    A = reshape(as, 2, N)
    B = reshape(bs, 2, N)

    return vcat(A,B)
end

function diverse_bcs(fitness::Function, iterations::Int=500, initial_candidates::Int=20)
    cands = []
    removenext = 1
    doremovenext = true
    incumbent_diversity_diff = 0
    for i in 1:iterations
        InitPopulation = draw_init_population(Class0Points, Class1Points) # FIXME generalize
        res = bboptimize(fitness; SearchRange = SearchRangePair,
                                    CallbackInterval = 0.0,
                                    CallbackFunction = mycallback,  # FIXME generalize
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

cands = diverse_bcs(fitness_function, 1000, 50)

using StatsPlots

as = map(c -> (c[1], c[2]), cands)
bs = map(c -> (c[3], c[4]), cands)

limits = (-20, 20)
plot(as, seriestype=:scatter, xlims=limits, ylims = limits)
plot!(bs, seriestype=:scatter)