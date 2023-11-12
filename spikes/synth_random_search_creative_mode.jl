using CSV, DataFrames
using BlackBoxOptim
using Statistics # mean

function argsmallest(A::AbstractArray{T,N}, n::Integer) where {T,N}
    # should someone ask more elements than array size, just sort array

    if n>= length(vec(A))
      ind=collect(1:length(vec(A)))
      ind=sortperm(A[ind])
      return CartesianIndices(A)[ind]
    end
    # otherwise 
    ind=collect(1:n)
    mymax=maximum(A[ind])
    for j=n+1:length(vec(A))
    if A[j]<mymax
     getout=findmax(A[ind])[2]
     ind[getout]=j
     mymax=maximum(A[ind])
    end
    end
    ind=ind[sortperm(A[ind])]
    
    return CartesianIndices(A)[ind]
end

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
global doremovenext = true
global incumbent_diversity_diff = 0
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

    #TODO adjust to add and remove points depending on overall metric (total length of distances!?).
    if length(cands) < 21
        push!(cands, cand)
    else
        #TODO if the new candidates local distance is larger than average, we dont remove next round
        if !doremovenext
            push!(cands, cands[removenext])
            global doremovenext = true
        end
        cands[removenext] = cand
        apoints = map(c -> c[1:2], cands)
        distances = pairwise(EuclideanDist, apoints)
        nearestneighbordistances = sort(distances, dims=2)[:,2:3]
        neighbordistancesums = sum(nearestneighbordistances, dims=2)[:,1]

        diversity_diff = sum(neighbordistancesums) / length(cands)

        furthestneighbors = nearestneighbordistances[:,2]
        #if mean(neighbordistancesums) < neighbordistancesums[removenext]
        if incumbent_diversity_diff < diversity_diff
            global incumbent_diversity_diff = diversity_diff
            #if mean(furthestneighbors) < furthestneighbors[removenext]
                global doremovenext = false
            #else
            #    global doremovenext = true
            #end
        end


        # TODO better pruning strategy, this one does not work well...
        # if i % 10 == 0
        #     #pruning - remove those being most squeezed (half of them) for cleansing purpose
        #     toberemoved = div(length(cands), 2)
        #     idxs = argsmallest(cands, toberemoved)
        #     global cands = cands[setdiff(1:end, idxs)]
        # else
            global removenext = argmin(neighbordistancesums) # remove closest to its two neighbor points (2d boundary has 2 neighbor points)
        # end
    end
    "************ $i" |> println
end

if doremovenext
    cands = cands[setdiff(1:end, removenext)] # ensure that the very last "removenext" is respected
end

using StatsPlots

as = map(c -> (c[1], c[2]), cands)
bs = map(c -> (c[3], c[4]), cands)
plot(as, seriestype=:scatter)
plot!(bs, seriestype=:scatter)