using CSV, DataFrames ,MLsqueeze, DecisionTree, Statistics

struct ExperimentPair
    id
    inputs
end

epairs = [ExperimentPair("synth", [:x1 , :x2]),
            ExperimentPair("iris", [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]),
            ExperimentPair("titanic", [:Pclass, :Age, :Sex, :Fare, :Parch, :SibSp]),
            ExperimentPair("car", [:buyingprice, :maintenancecost, :doors, :capacity, :luggageboot, :safety]),
            ExperimentPair("wine", [:Alcohol, :Malicacid, :Ash, :Acl, :Mg, :Phenols, :Flavanoids, :Nonflavanoidphenols, :Proanth, :Colorint, :Hue, :OD, :Proline]),
            ExperimentPair("heart", [:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang, :oldpeak, :slope,:ca, :thal]),
            ExperimentPair("adult", [:age,:fnlwgt,:educationalnum,:gender,:capitalgain,:capitalloss,:hoursperweek])
]

function pp_df_for(epair, n_cand, method)
    file = "data/expresults/$(epair.id)_bcs_$(method)_$(n_cand)_pp.csv"
    if !isfile(file)
        file = "data/expresults/$(epair.id)_bcs_$(method)_$(n_cand).csv"
    end

    return CSV.read(file, DataFrame)
end

function cal_distances(epair, n_cand, method, dir)
    files = filter(x -> occursin(r"^" * epair.id
                * r"_bcs_"
                * method
                * r"_"
                * string(n_cand)
                * r"_\d*.csv$", x), readdir(dir))

    nearestdists = zeros(Float64, 0,2)
    for f in files
        digs = digits(n_cand) |> length
        ppfile = f[1:end-(digs+4)] * "_pp.csv"
        if isfile(ppfile)
            f = ppfile
        end

        df = CSV.read(joinpath(dir, f), DataFrame)
        f_dists = two_nearest_neighbor_distances(df, epair.inputs)
        nearestdists = vcat(nearestdists, f_dists)
    end

    return sum(nearestdists, dims=2)
end

function boxplots_distances(epairs, dir = "data/expresults", n_cand = 20)
    local pl

    label_r = nothing
    label_d = nothing
    legend = false

    i::Int = 1
    for epair in epairs
        r_2nn = cal_distances(epair, n_cand, "random", dir)
        d_2nn = cal_distances(epair, n_cand, "div", dir)

        min = minimum([minimum(r_2nn), minimum(d_2nn)])
        max = maximum([maximum(r_2nn), maximum(d_2nn)])
        r_2nn = map(d -> (d - min) / (max - min), r_2nn)
        d_2nn = map(d -> (d - min) / (max - min), d_2nn)

        if i == length(epairs)
            label_r = "no div"
            label_d = "div"
            legend = :topright
        end

        if @isdefined(pl)
            violin!(pl, r_2nn; color=:red, legend, label = label_r)
        else
            pl = violin(r_2nn; color=:red, legend, label = label_r)
        end

        violin!(pl, d_2nn; color=:green, legend, label = label_d)
        i += 1
    end

    xlabels = map(e -> e.id, epairs)
    xticks_pos = map(x -> x + .5, 1:2:i*2)

    plot!(pl,xticks=(xticks_pos, xlabels), xrotation=45)
    xlabel!(pl, "Dataset")
    ylabel!(pl, "2-NN boundary candidate distance")

    return pl
end

# create a latex table head with columns dataset, method, number of candidates
function latex_table_head(io=stdout)
    println(io, "\\begin{tabular}{l|r|r|r}")
    println(io, "\\textbf{Dataset} & \\textbf{BC} & \\textbf{No Div} & \\textbf{Div} \\\\")
    println(io, "\\hline")
end

function latex_table_distances(epairs, io=stdout, dir="data/expresults")
    latex_table_head(io)
    for epair in epairs
        for n_cand in [10, 20]
            ds_r = cal_distances(epair, n_cand, "random", dir)
            ds_d = cal_distances(epair, n_cand, "div", dir)

            d_rand_nn_mean, d_rand_nn_sd = mean(ds_r), std(ds_r)
            d_div_nn_mean, d_div_nn_sd = mean(ds_d), std(ds_d)

            if n_cand == 10
                print(io, "\\textbf{$(epair.id)} ")
            end

            # change the order of method and candidates to make it more readable
            println(io, "& $(n_cand) & $(round(d_div_nn_mean, digits=2)) \$\\pm\$ $(round(d_div_nn_sd, digits=2)) & \\textbf{$(round(d_rand_nn_mean, digits=2))} \$\\pm\$ \\textbf{$(round(d_rand_nn_sd, digits=2))} \\\\")
        end
        println(io, "\\hline")
    end

    println(io, "\\end{tabular}")
end

open("data/distance_table.tex", "w") do io
    latex_table_distances(epairs, io)
 end

using StatsPlots
pl = boxplots_distances([epairs[1]])
Plots.svg(pl, "data/distance_violin")