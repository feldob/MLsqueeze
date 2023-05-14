using CSV, DataFrames
using BlackBoxOptim
# a search algorithm that creates a diverse set of boundary candidates.
# 1. import training dataset (binary, tableau, continuous inputs, 2d)
# 2. randomly sample input pairs from both categories (1/0).
# 3. use global search to "squeeze out" a boundary (requires definition of a "satisfactory" delta in the output).
#   - use Program Derivative for the squeeze in the fitness function, using standard EA search (diffevo).

DataDir = joinpath(@__DIR__(), "..", "data")
const DF = CSV.read(joinpath(DataDir, "synth.csv"), DataFrame)

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
synth_func(x) = (x+2)*(x^2-4)
check_synth_valid(x::Number,y::Number) = y > synth_func(x) ? 1 : 0

function fitness_function(x::Vector{Float64})
    classA = check_synth_valid(x[1], x[2])
    classB = check_synth_valid(x[3], x[4])
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

cands = []

global removenext = 1
for i in 1:1000
    InitPopulation = draw_init_population(Class0Points, Class1Points)
    res = bboptimize(fitness_function; SearchRange = SearchRangePair, 
                                CallbackInterval = 0.0,
                                CallbackFunction = mycallback,
                                MaxTime = .3,
                                Population = InitPopulation)
    cand = best_candidate(res)
    #TODO have a check for whether the search was successful... if points too far apart, not successf. also interesting - what are those cases? -> investigate.
    cand |> println

    if length(cands) < 41
        push!(cands, cand)
    else
        cands[removenext] = cand
        apoints = map(c -> c[1:2], cands)
        distances = pairwise(EuclideanDist, apoints)
        global removenext = argmin(sum(sort(distances, dims=2)[:,2:3], dims=2)[:,1]) # remove closest to its two neighbor points (2d boundary has 2 neighbor points)
    end
    "************ $i" |> println
end

cands = cands[setdiff(1:end, removenext)] # ensure that the very last "removenext" is respected

using StatsPlots

as = map(c -> (c[1], c[2]), cands)
bs = map(c -> (c[3], c[4]), cands)
plot(as, seriestype=:scatter)
plot!(bs, seriestype=:scatter)