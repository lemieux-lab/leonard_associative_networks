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


function stringify(p::Dict;spacer = 80)  
    s = join(["$key: $val" for (key, val) in p], ", ")
    for i in collect(spacer:spacer:length(s))
        s = "$(s[1:i])\n$(s[i:end])"
    end
    return s 
end 

##########################################
####### Plotting functions    ############
##########################################

function plot_embed(X_tr, labels, assoc_ae_params,fig_outpath)
    # plot final 2d embed from Auto-Encoder
    tr_acc = round(assoc_ae_params["tr_acc"], digits = 3) * 100
    embed = DataFrame(:emb1=>X_tr[1,:], :emb2=>X_tr[2,:], :cancer_type => labels)
    p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type)
    fig = draw(p, axis = (;width = 1224, height = 1024, 
    title="$(assoc_ae_params["model_type"]) on $(assoc_ae_params["dataset"]) data\naccuracy by DNN : $tr_acc%"))
    CairoMakie.save(fig_outpath, fig)
end 

function plot_learning_curves(learning_curves, assoc_ae_params, fig_outpath)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :ae_loss=>[i[1] for i in learning_curves], :ae_cor => [i[2] for i in learning_curves],
    :clf_loss=>[i[3] for i in learning_curves], :clf_acc => [i[4] for i in learning_curves])
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss"], color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"clf_loss"])
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder Pearson Corr.")
    ae_loss = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor"], color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    ae_loss = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"clf_acc"] .* 100 )
    Label(fig[3,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(assoc_ae_params))")
    CairoMakie.save(fig_outpath, fig)
end 