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

function cal_distances(epair, n_cand, method)
    file = "data/expresults/$(epair.id)_bcs_$(method)_$(n_cand)_pp.csv"
    if !isfile(file)
        file = "data/expresults/$(epair.id)_bcs_$(method)_$(n_cand).csv"
    end

    df = CSV.read(file, DataFrame)
    return two_nearest_neighbor_distances(df, epair.inputs)
end

# create a latex table head with columns dataset, method, number of candidates
function latex_table_head()
    println("\\begin{tabular}{l|r|r|r}")
    println("\\textbf{Dataset} & \\textbf{Candidates} & \\textbf{Method} & \\textbf{Distance} \\\\")
    println("\\hline")
end

latex_table_head()


for epair in epairs
    for n_cand in [10, 20]
        d_rand_nn_mean, d_rand_nn_sd = cal_distances(epair, n_cand, "random")
        d_div_nn_mean, d_div_nn_sd = cal_distances(epair, n_cand, "div")

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