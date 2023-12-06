using CSV, DataFrames ,MLsqueeze, DecisionTree

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

function cal_distances(epair, n_cand, method)
    df =pp_df_for(epair, n_cand, method)
    nearestdists = two_nearest_neighbor_distances(df, epair.inputs)
    return sum(nearestdists, dims=2)
end

function cal_distances_stats(epair, n_cand, method)
    df =pp_df_for(epair, n_cand, method)
    return two_nearest_neighbor_distances_stats(df, epair.inputs)
end

function boxplots_distances(epairs, n_cand = 10)
    local pl

    label_r = nothing
    label_d = nothing
    legend = false

    i::Int = 1
    for epair in epairs
        r_2nn = cal_distances(epair, n_cand, "random")
        d_2nn = cal_distances(epair, n_cand, "div")

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
function latex_table_head()
    println("\\begin{tabular}{l|r|r|r}")
    println("\\textbf{Dataset} & \\textbf{Candidates} & \\textbf{Method} & \\textbf{Distance} \\\\")
    println("\\hline")
end

function latex_table_distances(epairs)
    latex_table_head()
    for epair in epairs
        for n_cand in [10, 20]
            d_rand_nn_mean, d_rand_nn_sd = cal_distances_stats(epair, n_cand, "random")
            d_div_nn_mean, d_div_nn_sd = cal_distances_stats(epair, n_cand, "div")

            if n_cand == 10
                print("\\textbf{$(epair.id)} ")
            end

            # change the order of method and candidates to make it more readable
            println("& $(n_cand) & div & $(round(d_div_nn_mean, digits=2)) ± $(round(d_div_nn_sd, digits=2)) \\\\")
            println("& $(n_cand) & random & $(round(d_rand_nn_mean, digits=2)) ± $(round(d_rand_nn_sd, digits=2)) \\\\")
        end
        println("\\hline")
        #TODO add LaTeX table output
    end

    # print latex table footer that ends the tabular
    println("\\end{tabular}")
end

#latex_table_distances(epairs)

using StatsPlots
pl = boxplots_distances(epairs)