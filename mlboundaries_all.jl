using MLsqueeze, CSV, DataFrames, DecisionTree

struct BoundaryExperiment
    name
    df
    inputs
    output
    rounds

    BoundaryExperiment(name, file, inputs, output, rounds = 10) = new(name, CSV.read(file, DataFrame), inputs, output, rounds)
end

inputstotal(exp::BoundaryExperiment) = ncol(exp.df)-1
inputsused(exp::BoundaryExperiment) = length(exp.inputs)
outputalternatives(exp::BoundaryExperiment) = length(unique(exp.df[!, exp.output]))
rounds(exp::BoundaryExperiment) = exp.rounds

synth_exp = BoundaryExperiment("synth", "data/synth.csv", [:x1, :x2], :class)
iris_exp = BoundaryExperiment("iris", "data/Iris.csv", [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm], :Species)
heart_exp = BoundaryExperiment("heart", "data/heart.csv", [:age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang, :oldpeak, :slope,:ca, :thal], :target)
wine_exp = BoundaryExperiment("wine", "data/wine.csv", [:Alcohol,:Malicacid,:Ash,:Acl,:Mg,:Phenols,:Flavanoids,:Nonflavanoidphenols,:Proanth,:Colorint,:Hue,:OD,:Proline], :Wine)
titanic_exp = BoundaryExperiment("titanic", "data/titanic_pp.csv", [:Pclass,:Sex,:Age,:SibSp,:Parch,:Fare], :Survived)
car_exp = BoundaryExperiment("car", "data/car_evaluation_pp.csv", [:buyingprice,:maintenancecost,:doors,:capacity,:luggageboot,:safety], :decision)
adult_exp = BoundaryExperiment("adult", "data/adult_pp.csv", [:age,:fnlwgt,:educationalnum,:gender,:capitalgain,:capitalloss,:hoursperweek], :income)

exps = [ adult_exp, car_exp, titanic_exp, wine_exp, synth_exp, iris_exp, heart_exp ]

for exp in exps
    td = TrainingData(exp.name, exp.df; inputs=exp.inputs, output=exp.output)
    modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
    be = BoundaryExposer(td, modelsut)
    
    for n_cand in [10, 20]
        for r in rounds
            candidates = apply(be; iterations =n_cand, initial_candidates=10, optimizefordiversity=false)
            df = todataframe(candidates, modelsut; output = exp.output)
            CSV.write("data/expresults/$(exp.name)_bcs_random_$(n_cand)_$r.csv", df)

            candidates = apply(be; iterations=2000, initial_candidates=n_cand, add_new=false)
            df = todataframe(candidates, modelsut; output = exp.output)
            CSV.write("data/expresults/$(exp.name)_bcs_div_$(n_cand)_$r.csv", df)
        end
    end
end

# -----------------------    LATEX TABLE        ----------------------------------------------

# println("\\begin{tabular}{l|c|c|c}")
# println("\\textbf{Dataset} & \\textbf{Features} & \\textbf{Used Features} & \\textbf{Output} \\\\")

# for exp in exps
#     println("\\textbf{$(exp.name)} & $(inputstotal(exp)) & $(inputsused(exp)) & $(outputalternatives(exp)) \\\\")
# end