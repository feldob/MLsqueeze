include("bmi_sut.jl")

using StatsPlots

#plot(x, y, xlims=(-30, 250), ylims=(-30,530), label = "boundary")

#TODO adjust range to BMI problem
randin_height_range(n::Integer) = -10 .+ rand(n) .* 243
randin_weight_range(n::Integer) = -10 .+ rand(n) .* 510

using DataFrames

npoints = 1000
df = DataFrame()
df.x1 = randin_height_range(npoints)
df.x2 = randin_weight_range(npoints)

df.class = map(r -> bmi_classification(r.x1, r.x2), eachrow(df))

gfs = groupby(df, :class)
p = plot(gfs[1].x1,gfs[1].x2, xlims = (-30, 250), ylims=(-30,530), seriestype=:scatter, label = first(gfs[1].class))

for i in 2:length(gfs)
    plot!(p, gfs[i].x1,gfs[i].x2, seriestype=:scatter, label = first(gfs[i].class))
end

# ## to print plot
png(p, "data/bmi_class.png")

using DecisionTree

for depth in 2:5
    model = DecisionTreeClassifier(max_depth=depth)
    training = hcat(df.x1, df.x2)
    test = training # OBS here using same data for training and testing
    DecisionTree.fit!(model, training, df.class)
    df[!, "DT_d$(depth)"]=DecisionTree.predict(model, test)
end


## to export to CSV file
using CSV
CSV.write("data/bmi.csv", df)