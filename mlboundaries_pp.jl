using CSV, DataFrames

dir = "data/expresults/"

function convert_to_pp(dataset_id, conv_func!)
    for file in filter(f -> startswith(f, dataset_id) && !endswith(f, "_pp.csv"), readdir(dir))
        df = CSV.read(joinpath(dir, file), DataFrame)
        conv_func!(df)
        pp_file = joinpath(dir, file[1:end-4] * "_pp.csv")
        CSV.write(pp_file, df)
    end
end

# ----- adult
function adult_conv_func!(df)
    df.gender = round.(df.gender)
    df.n_gender = round.(df.n_gender)
end

# ----- titanic
function titanic_conv_func!(df)
    df.Pclass = round.(Int, df.Pclass)
    df.n_Pclass = round.(Int, df.n_Pclass)
    df.Age = round.(Int, df.Age)
    df.n_Age = round.(Int, df.n_Age)
    df.SibSp = round.(Int, df.SibSp)
    df.n_SibSp = round.(Int, df.n_SibSp)
    df.Parch = round.(Int, df.Parch)
    df.n_Parch = round.(Int, df.n_Parch)
    df.Fare = round.(df.Fare, digits=2)
    df.n_Fare = round.(df.n_Fare, digits=2)
    df.Sex = round.(df.Sex)
    df.n_Sex = round.(df.n_Sex)
end

# ----- car evaluation
function car_conv_func!(df)
    df.buyingprice = round.(Int, df.buyingprice)
    df.n_buyingprice = round.(Int, df.n_buyingprice)
    df.maintenancecost = round.(Int, df.maintenancecost)
    df.n_maintenancecost = round.(Int, df.n_maintenancecost)
    df.doors = round.(Int, df.doors)
    df.n_doors = round.(Int, df.n_doors)
    df.capacity = round.(Int, df.capacity)
    df.n_capacity = round.(Int, df.n_capacity)
    df.luggageboot = round.(Int, df.luggageboot)
    df.n_luggageboot = round.(Int, df.n_luggageboot)
    df.safety = round.(Int, df.safety)
    df.n_safety = round.(Int, df.n_safety)
end

# run
convert_to_pp("adult", adult_conv_func!)
convert_to_pp("titanic", titanic_conv_func!)
convert_to_pp("car", car_conv_func!)