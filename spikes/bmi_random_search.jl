include("random_search.jl")
include("bmi_sut.jl")

using CSV, DataFrames

DataDir = joinpath(@__DIR__(), "..", "data")

#To avoid using doubles, we receive an integer height in CM!

DF = CSV.read(joinpath(DataDir, "bmi.csv"), DataFrame)

# Inputs... then output
O = "class"
oidx = findfirst(==(O), names(DF))
Inps = names(DF)[1:(oidx-1)]

#TODO check_bmi_valid should be a check for the class, and the pd function should calculate distance based on either ordinal or one-hot-coding as a choice.

#DF.class = map(r -> check_bmi_valid(r.x1, r.x2) ? 1 : 0, eachrow(DF))
SearchRangePair, Points = config_search(DF, :class)

cands = diverse_bcs(pd_fitness(bmi_classification), Points, 1000, 20, SearchRangePair)

using StatsPlots

function plot_naive(cands)
    as = map(c -> (c[1], c[2]), cands)
    bs = map(c -> (c[3], c[4]), cands)
    plot(as, xlabel = "heigth (cm)", ylabel = "weight (kg)", seriestype=:scatter, label="boundaries BMI")
    #plot!(bs, seriestype=:scatter, label="normal")
end

function plot_colored_boundary(cands)
    as = map(c -> (c[1], c[2]), cands)
    bs = map(c -> (c[3], c[4]), cands)
    as_c = map(a -> bmi_classification(a[1], a[2]), as)
    bs_c = map(b -> bmi_classification(b[1], b[2]), bs)

    ids = map(i -> join(sort([as_c[i], bs_c[i]])), eachindex(as))

    idlookup = Dict()
    map(id -> idlookup[id] = Set(), unique(ids))
    foreach(i -> push!(idlookup[ids[i]], i) , eachindex(ids))

    p = nothing
    first = true
    for type in keys(idlookup)
        a = as[[values(idlookup[type])...]]
        if first
            p = plot(a, xlabel = "heigth (cm)", ylabel = "weight (kg)", seriestype=:scatter, label=type)
            first = false
        end
        plot!(p, a, seriestype=:scatter, label=type)
    end

    png(p, "data/bmi_colored_class.png")

end

plot_naive(cands)
plot_colored_boundary(cands)

# TODO check the boundary for each point and assign color.
