# dump folds 
function dump_folds(folds, params::Dict, case_ids)
    f = h5open("RES/$(params["session_id"])/$(params["modelid"])/fold_ids.bson", "w")
    test_ids = Array{String, 2}(undef, (length(folds), length(folds[1]["test_ids"])))
    [test_ids[i,:] = case_ids[folds[i]["test_ids"]] for i in 1:length(folds)]
    f["test_ids"] =  test_ids
    train_ids = Array{String, 2}(undef, (length(folds), length(folds[1]["train_ids"])))
    [train_ids[i,:] = case_ids[folds[i]["train_ids"]] for i in 1:length(folds)]
    f["train_ids"] = train_ids 
    close(f)
end
dump_folds(folds, assoc_ae_params, tcga_prediction.rows)
function load_folds(params)
    # fold ids loading 
    inf = h5open("RES/$(params["session_id"])/$(params["modelid"])/fold_ids.bson", "r")
    test_ids = inf["test_ids"][:,:]
    train_ids = inf["train_ids"][:,:]
    close(inf)
    return train_ids, test_ids
end 

get_fold_ids(foldn, ids, case_ids) = findall([in(r, ids[foldn,:]) for r in case_ids])
get_fold_data(foldn, ids, cdata) = cdata.data[find_fold_ids(foldn, ids, cdata.rows),:]

############################
###### General utilities ###
############################
zpad(n::Int;pad::Int=9) = lpad(string(n),pad,'0')