using GLPK
using HydroPowerModels

#Simulating the complete horizon
testcases_dir = joinpath(dirname(dirname(dirname(@__FILE__))), "testcases")
alldata = HydroPowerModels.parse_folder(joinpath(testcases_dir, "case3deterministic horizon"))
params = create_param(;
    stages=60,
    model_constructor_grid=DCPPowerModel,
    post_method=PowerModels.build_opf,
    optimizer=GLPK.Optimizer,
    discount_factor=0.995,
);
m = hydro_thermal_operation(alldata, params)
HydroPowerModels.train(m; iteration_limit=60);
results = HydroPowerModels.simulate(m, 1);

#Simulating from the second year on (tini = 13)
for data in alldata
    data["hydro"]["Hydrogenerators"][1]["initial_volume"] = 0.1229422863405989
end
params = create_param(;
    stages=48,
    model_constructor_grid=DCPPowerModel,
    post_method=PowerModels.build_opf,
    optimizer=GLPK.Optimizer,
    discount_factor=0.995,
);
m = hydro_thermal_operation(alldata, params)
HydroPowerModels.train(m; iteration_limit=60);
results = HydroPowerModels.simulate(m, 1);

#Simulating shorter horizon with no opporty cost 
testcases_dir = joinpath(dirname(dirname(dirname(@__FILE__))), "testcases")
alldata = HydroPowerModels.parse_folder(joinpath(testcases_dir, "case3deterministic horizon"))
for data in alldata
    data["hydro"]["Hydrogenerators"][1]["final_volume"] = 0.464942
end
params = create_param(;
    stages=3,
    model_constructor_grid=DCPPowerModel,
    post_method=PowerModels.build_opf,
    optimizer=GLPK.Optimizer,
    discount_factor=0.995,
);
m = hydro_thermal_operation(alldata, params)
HydroPowerModels.train(m; iteration_limit=60);
results = HydroPowerModels.simulate(m, 1);

#Simulating the implementation of a policy with K-steps of look-ahead
testcases_dir = joinpath(dirname(dirname(dirname(@__FILE__))), "testcases")
alldata = HydroPowerModels.parse_folder(joinpath(testcases_dir, "case3deterministic horizon"))

p = deepcopy(alldata[1]["hydro"]["scenario_probabilities"])
v = deepcopy(alldata[1]["hydro"]["Hydrogenerators"][1]["inflow"])

t=1

for t = 1:59
    Tfim = min(59,t+59)
    Ksteps = Tfim-t+1

    for data in alldata
        data["hydro"]["scenario_probabilities"] = p[t:end,:]
        data["hydro"]["Hydrogenerators"][1]["inflow"] = v[t:end,:] * 1.5
        data["hydro"]["Hydrogenerators"][1]["inflow"][1,1] = data["hydro"]["Hydrogenerators"][1]["inflow"][1,1] / 1.5
    end

    params = create_param(;
    stages=Ksteps,
    model_constructor_grid=DCPPowerModel,
    post_method=PowerModels.build_opf,
    optimizer=GLPK.Optimizer,
    discount_factor=0.995,
    );
    m = hydro_thermal_operation(alldata, params)
    HydroPowerModels.train(m; iteration_limit=60);
    results = HydroPowerModels.simulate(m, 1);

    for data in alldata
        data["hydro"]["Hydrogenerators"][1]["final_volume"] = results[:simulations][1][end][:reservoirs][:reservoir][1].out
    end
end
