const hydro_schema_path = joinpath(dirname(@__FILE__), "jsonschema", "hydro.json")
const powermodels_schema_path = joinpath(
    dirname(@__FILE__), "jsonschema", "PowerModels.json"
)

"""
Read hydro description json file.
"""
function parse_file_json(file::String)
    return JSON.parse(String(read(file)))
end

"""Read Hydrogenerators inflow csv file"""
function read_inflow(file::String, nHyd::Int)
    allinflows = CSV.read(file, Tables.matrix; header=false)
    nlin, ncol = size(allinflows)
    nCen = Int(floor(ncol / nHyd))
    vector_inflows = Array{Array{Float64,2}}(undef, nHyd)
    for i in 1:nHyd
        vector_inflows[i] = allinflows[1:nlin, ((i - 1) * nCen + 1):(i * nCen)]
    end
    return vector_inflows, nCen
end

"""
    HydroPowerModels.parse_folder(folder::String; stages::Int = 1,digts::Int=7)

Read hydrothermal case folder.

Parameters:
-   folder  : Path to case folder.
-   stages  : Number of stages.
-   digts   : Number of digits to take into acoint from case description files.
"""
function parse_folder(folder::String; stages::Int=1, digts::Int=7)
    data = Dict()
    try
        data["powersystem"] = parse_file_json(joinpath(folder, "PowerModels.json"))
        if typeof(data["powersystem"]["source_version"]) <: AbstractDict
            data["powersystem"]["source_version"] = VersionNumber(
                data["powersystem"]["source_version"]["major"],
                data["powersystem"]["source_version"]["minor"],
                data["powersystem"]["source_version"]["patch"],
                Tuple{}(data["powersystem"]["source_version"]["prerelease"]),
                Tuple{}(data["powersystem"]["source_version"]["build"]),
            )
        end
    catch
        data["powersystem"] = PowerModels.parse_file(joinpath(folder, "PowerModels.m"))
    end
    data["hydro"] = parse_file_json(joinpath(folder, "hydro.json"))
    for i in 1:length(data["hydro"]["Hydrogenerators"])
        data["hydro"]["Hydrogenerators"][i] = signif_dict(
            data["hydro"]["Hydrogenerators"][i], digts
        )
    end
    vector_inflows, nCen = read_inflow(
        joinpath(folder, "inflows.csv"), length(data["hydro"]["Hydrogenerators"])
    )
    for i in 1:length(data["hydro"]["Hydrogenerators"])
        data["hydro"]["Hydrogenerators"][i]["inflow"] = vector_inflows[i]
    end
    try
        data["hydro"]["scenario_probabilities"] = convert(
            Matrix{Float64},
            CSV.read(joinpath(folder, "scenarioprobability.csv"); header=false),
        )
    catch
        data["hydro"]["scenario_probabilities"] =
            ones(size(vector_inflows[1], 1), nCen) ./ nCen
    end
    return [deepcopy(data) for _ in 1:stages]
end

"""set active demand"""
function set_active_demand!(alldata::Array{Dict{Any,Any}}, demand::Array{Float64,2})
    for t in 1:size(alldata, 1)
        data = alldata[t]
        for load in 1:length(data["powersystem"]["load"])
            bus = data["powersystem"]["load"]["$load"]["load_bus"]
            data["powersystem"]["load"]["$load"]["pd"] = demand[t, bus]
        end
    end
    return nothing
end

"""
    create_param(;stages::Int = 1,
        model_constructor_grid::Type = DCPPowerModel,
        model_constructor_grid_backward::Type = model_constructor_grid,
        model_constructor_grid_forward::Type = model_constructor_grid_backward,
        post_method::Function = PowerModels.build_opf,
        optimizer::DataType = GLPK.Optimizer,
        optimizer_backward::DataType = optimizer,
        optimizer_forward::DataType = optimizer_backward,
        setting::Dict = Dict("output" => Dict("branch_flows" => true,"duals" => true)),
        verbose::Bool = false,
        stage_hours::Int = 1,
        discount_factor::Float64 = 1.0,
        cycle_probability::Float64 = 0.0)

Create Parameters Dictionary.

Keywords are:
-   stages::Int                      : Number of stages.
-   model_constructor_grid           : Default Network formulation (Types from <https://github.com/lanl-ansi/PowerModels.jl>).
-   model_constructor_grid_backward  : Network formulation used in backward (Types from <https://github.com/lanl-ansi/PowerModels.jl>).
-   model_constructor_grid_forward   : Network formulation used in forward (Types from <https://github.com/lanl-ansi/PowerModels.jl>).
-   post_method                      : The post method.
-   optimizer                        : Default optimizer factory (<http://www.juliaopt.org/JuMP.jl/v0.19.0/solvers/>).
-   optimizer_backward               : Optimizer factory used in backward (<http://www.juliaopt.org/JuMP.jl/v0.19.0/solvers/>).
-   optimizer_forward                : Optimizer factory used in forward(<http://www.juliaopt.org/JuMP.jl/v0.19.0/solvers/>).
-   setting                          : PowerModels settings (<https://github.com/lanl-ansi/PowerModels.jl/blob/e28644bf85232a5322adeeb847c0d18b7ff4f235/src/core/base.jl#L6-L34>)) .
-   verbose                          : Boolean to indicate information prints.
-   stage_hours                      : Number of hours in each stage.
-   discount_factor                  : The discount factor.
-   cycle_probability                : Probability of restart the horizon.
"""
function create_param(;
    stages::Int=1,
    model_constructor_grid::Type=DCPPowerModel,
    model_constructor_grid_backward::Type=model_constructor_grid,
    model_constructor_grid_forward::Type=model_constructor_grid_backward,
    post_method::Function=PowerModels.build_opf,
    optimizer::DataType=GLPK.Optimizer,
    optimizer_backward::DataType=optimizer,
    optimizer_forward::DataType=optimizer_backward,
    setting::Dict=Dict("output" => Dict("branch_flows" => true, "duals" => true)),
    verbose::Bool=false,
    stage_hours::Int=1,
    discount_factor::Float64=1.0,
    cycle_probability::Float64=0.0,
)
    params = Dict()
    params["stages"] = stages
    params["stage_hours"] = stage_hours
    params["discount_factor"] = discount_factor
    params["cycle_probability"] = cycle_probability
    params["model_constructor_grid"] = model_constructor_grid
    params["model_constructor_grid_backward"] = model_constructor_grid_backward
    params["model_constructor_grid_forward"] = model_constructor_grid_forward
    params["post_method"] = post_method
    params["optimizer"] = optimizer
    params["optimizer_backward"] = optimizer_backward
    params["optimizer_forward"] = optimizer_forward
    params["verbose"] = verbose
    params["setting"] = setting
    return params
end

"""
    HydroPowerModels.validate_json(json_path::String, schema_path::String)

Returns nothing if the json 'json_path' matchs the schema 'json_path', showns the errors otherwise.

Parameters:
-   json_path   : Path to json.
-   schema_path : Path to schema.
"""
function validate_json(json_path::String, schema_path::String)
    return validate(Schema(parse_file_json(schema_path)), parse_file_json(json_path))
end

"""
    HydroPowerModels.validate_json_hydro(json_path::String)

Returns nothing if the json 'json_path' matchs the hydro schema, showns the errors otherwise.

Parameters:
-   json_path   : Path to json.
"""
function validate_json_hydro(json_path::String)
    return validate_json(json_path, hydro_schema_path)
end

"""
    HydroPowerModels.validate_json_powermodels(json_path::String)

Returns nothing if the json 'json_path' matchs the powermodels schema, showns the errors otherwise.

Parameters:
-   json_path   : Path to json.
"""
function validate_json_powermodels(json_path::String)
    return validate_json(json_path, powermodels_schema_path)
end
