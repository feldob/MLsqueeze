using CSV, DataFrames
using BlackBoxOptim
using Dates

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

# Calculate the closest other-class point, either exactly or by sampling
# a fixed number of points from the other class, for each point in the dataset.
function closest_otherclass_points(inputmatrix, outputs; distance = EuclideanDist, numsamples = 0)
    numpoints = length(outputs)
    @assert numpoints == size(inputmatrix, 1)
    result = []
    for i in 1:numpoints
        otherclassidxs = findall(x -> x != outputs[i], outputs)
        # Find the closest point in the other class
        min_dist, min_j = Inf, -1
        idxs = (numsamples < 1 || numsamples >= length(otherclassidxs)) ? 
            otherclassidxs : 
            sample(otherclassidxs, numsamples, replace = false)
        for j in idxs
            d = distance(inputmatrix[i, :], inputmatrix[j, :])
            if d < min_dist
                min_dist, min_j = d, j
            end
        end
        push!(result, (i, min_j, min_dist))
    end
    sort!(result, by = last)
    return result
end

closestpairs = closest_otherclass_points(Inputs, Outputs)
startAindex, startBindex, d = closestpairs[1]

# Fitness 1a is to be close to B but same class as A
# Fitness 1b is to be close to A but same class as B
# Penalize heavily if switching class
function fitness_close_but_same_class(newpoint, newclass, targetpoint, targetclass, distance = EuclideanDist)
    d = distance(newpoint, targetpoint)
    return (newclass != targetclass) ? (10_000+d) : d
end

function old_fitness1(XYs, startA, startB, classA, classB, modelfunc, dims=2)
    fitness = 0.0
    # First point in each pair should have classA but be close to startB
    for i in 1:dims:(length(XYs)-1)
        newpoint = view(XYs, i:(i+dims-1))
        newclass = modelfunc(newpoint)
        fitness += fitness_close_but_same_class(newpoint, newclass, startB, classA)
    end
    # Second point in each pair should have classB but be close to startA
    for i in 2:dims:(length(XYs)-1)
        newpoint = view(XYs, i:(i+dims-1))
        newclass = modelfunc(newpoint)
        fitness += fitness_close_but_same_class(newpoint, newclass, startA, classB)
    end
    return fitness
end

# Fitness 2: calculate the pairwise distance matrix between the first
# points of each pair and then sum the distances to the closest point.
function fitness2(XYs, dims=2)
    # Negate since we want to maximize
    return -(sum_closest_per_point(XYs, 1; dims) + sum_closest_per_point(XYs, 2; dims))
end

function sum_closest_per_point(XYs, startidx; distance = EuclideanDist, dims = 2)
    numpairs = div(length(XYs), 2*dims)
    mindists = Inf .* ones(Float64, Int(numpairs))
    for i in 1:numpairs
        pi = 1 + dims * ((startidx-1) + (i-1)*2)
        newpoint = view(XYs, pi:(pi+dims-1))
        for j in (i+1):numpairs
            pj = 1 + dims * ((startidx-1) + (j-1)*2)
            otherpoint = view(XYs, pj:(pj+dims-1))
            d = distance(newpoint, otherpoint)
            if d < mindists[i]
                mindists[i] = d
            end
            if d < mindists[j]
                mindists[j] = d
            end
        end
    end
    return sum(mindists)
end

# Just use a global one for now, for simplicity
BestFitness = Inf

function find_boundary_set(startA, startB, classA, classB, modelfunc; 
    N = 10, fitness1weight = 1, fitness2weight = 100, kwargs...)
    # We are searching for N pairs of points, i.e. 2N points in total
    # Old Fitness 1: (2*N components)
    #   In each pair, the first point should be as close as possible to startB but have classA
    #   In each pair, the second point should be as close as possible to startA but have classB
    #
    # Fitness 1, BoundaryPairs: (not yet implemented, old fitness 1 for now)
    #   Fitness 1a, PairPointsDifferInMeaning: (N components) Points in a pair have different classes (outputs)
    #     (The generalization here is that one of the outputs in the pair should be close to A and one close to B)
    #   Fitness 1b, PairPointsAreClose: (N components) Points in a pair are close to each other (genotypically, inputs)
    #
    # Fitness 2, DiversePairs: (2*N components)
    #   A pair is far from its closest other pair
    #
    # Maybe have this instead of old 1, but I'm not sure why we need it. Maybe for more complex cases where we want to stay close to starting point or we might be lost in the output space!?
    # Fitness 3, CloseToStart: (2*N components)
    #   Pairs keep within a certain distance of the starting points
    
    # Now setup the dataframe we will add to as we find better fitnesses
    df = DataFrame(Time = DateTime[], 
            Ax = Float64[], Ay = Float64[], Bx = Float64[], By = Float64[])
    dims = length(startA)
    for i in 1:N
        df[!, "candA$(i)x"] = Float64[]
        df[!, "candA$(i)y"] = Float64[]
        df[!, "candB$(i)x"] = Float64[]
        df[!, "candB$(i)y"] = Float64[]
    end
    df[!, "fitness1"] = Float64[]
    df[!, "fitness2"] = Float64[]
    df[!, "fitness3"] = Float64[]

    fitness_function(XYs::Vector{Float64}) = begin
        fit1 = old_fitness1(XYs, startA, startB, classA, classB, modelfunc)
        fit2 = fitness2(XYs)
        fitness = fitness1weight * fit1/(2*N) + fitness2weight * fit2/(2*N)
        global BestFitness
        if fitness < BestFitness
            meta = [now(), startA[1], startA[2], startB[1], startB[2]]
            fitnesses = [fit1, fit2, fitness]
            # We always save the new best fitness, but if fitness difference 
            # is below 1e-4 we delete previous one with some probability.
            # This way we don't need to save so many of them when there are 
            # plateaus or convergence.
            fitnessdiff = abs(BestFitness - fitness)
            if fitnessdiff < 1e-4 && rand() < (fitnessdiff/1e-4)
                delete!(df, nrow(df))
            end
            push!(df, vcat(meta, XYs, fitnesses))
            BestFitness = fitness
        end
        return fitness
    end

    searchrange = vcat(map(_ -> SearchRangePair, 1:N)...)
    res = bboptimize(fitness_function; SearchRange = searchrange, kwargs...)

    return res, df
end

# Ok, now let's run this for the closest pair:
startA = Inputs[startAindex, :]
startB = Inputs[startBindex, :]
synth_func(x) = (x+2)*(x^2-4)
check_synth_valid(x::Number,y::Number) = y > synth_func(x) ? 1 : 0
synth_model_func(x) = check_synth_valid(x[1], x[2])
classA = synth_model_func(startA)
classB = synth_model_func(startB)
modelfunc = synth_model_func
f1w = 5
f2w = 1
res, df = find_boundary_set(startA, startB, classA, classB, modelfunc; 
    N = 10, fitness1weight = f1w, fitness2weight = f2w, 
    MaxTime = 30.0, PopulationSize = 500)
df |> CSV.write(joinpath(@__DIR__(), "synth_boundary_set_w$(f1w)_w$f2w.csv"))