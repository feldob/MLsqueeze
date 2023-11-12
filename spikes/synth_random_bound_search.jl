include("random_search.jl")

using CSV, DataFrames

DataDir = joinpath(@__DIR__(), "..", "data")
DF = CSV.read(joinpath(DataDir, "synth_bound.csv"), DataFrame)

# Inputs... then output
O = "class"
oidx = findfirst(==(O), names(DF))
Inps = names(DF)[1:(oidx-1)]

SearchRangePair, Points = config_search(DF)

# fitness function: the format is floats... four floats, class is a consequence only.
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

cands = diverse_bcs(pd_fitness(check_synth_bound_valid), Points, 1000, 50, SearchRangePair)

using StatsPlots

as = map(c -> (c[1], c[2]), cands)
bs = map(c -> (c[3], c[4]), cands)

limits = (-20, 20)
plot(as, seriestype=:scatter, xlims=limits, ylims = limits)
plot!(bs, seriestype=:scatter)