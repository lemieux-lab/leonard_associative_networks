
struct dnn
    model::Flux.Chain 
    opt
    lossf
end
function dnn(params::Dict)
    mdl_chain = gpu(Flux.Chain(
    Flux.Dense(params["dim_redux"], params["clf_hl_size"], relu), 
    Flux.Dense(params["clf_hl_size"], params["clf_hl_size"], relu), 
    Flux.Dense(params["clf_hl_size"], params["nclasses"], identity)))
    mdl_opt = Flux.ADAM(params["lr"])
    mdl_lossf = crossentropy_l2
    mdl = dnn(mdl_chain, mdl_opt, mdl_lossf)
    return mdl
end 

struct logistic_regression
    model::Flux.Dense
    opt
    lossf
end 

struct FE_model
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
    opt
    lossf
end

function FE_model(params::Dict)
    emb_size_1 = params["emb_size_1"]
    emb_size_2 = params["emb_size_2"]
    a = emb_size_1 + emb_size_2 
    b, c = params["fe_hl1_size"], params["fe_hl2_size"] 
    emb_layer_1 = gpu(Flux.Embedding(params["nsamples"], emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(params["ngenes"], emb_size_2))
    hl1 = gpu(Flux.Dense(a, b, relu))
    hl2 = gpu(Flux.Dense(b, c, relu))
    outpl = gpu(Flux.Dense(c, 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    opt = Flux.ADAM(params["lr"])
    lossf = mse_l2
    FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl, opt, lossf)
end 

struct assoc_FE
    fe::FE_model
    fe_data::DataFE
    clf::dnn 
end 

struct AE_model
    net::Flux.Chain 
    encoder::Flux.Chain
    decoder::Flux.Chain
    outpl::Flux.Dense
    opt
    lossf
end 

function AE_model(params::Dict)
    ## 2 x 2 Hidden layers Auto-Encoder model architecture.  
    enc_hl1 = gpu(Flux.Dense(params["ngenes"], params["enc_hl_size"], relu))
    enc_hl2 = gpu(Flux.Dense(params["enc_hl_size"], params["enc_hl_size"], relu))

    redux_layer = gpu(Flux.Dense(params["enc_hl_size"], params["dim_redux"], identity))
    
    dec_hl1 = gpu(Flux.Dense(params["dim_redux"], params["dec_hl_size"], relu))
    dec_hl2 = gpu(Flux.Dense(params["dec_hl_size"], params["dec_hl_size"], relu))

    outpl = gpu(Flux.Dense(params["dec_hl_size"], params["ngenes"], identity))

    net = gpu(Flux.Chain(
        enc_hl1, enc_hl2, redux_layer, dec_hl1, dec_hl2, outpl    
    ))
    encoder = gpu(Flux.Chain(enc_hl1, enc_hl2, redux_layer))
    decoder = gpu(Flux.Chain(dec_hl1, dec_hl2, outpl))

    opt = Flux.ADAM(params["lr"])
    lossf = mse_l2
    AE_model(net, encoder, decoder, outpl, opt, lossf)
end 

struct assoc_AE
    ae::AE_model 
    clf::dnn
end 

function l2_penalty(model::logistic_regression)
    return sum(abs2, model.model.weight)
end 

function l2_penalty(model::dnn)
    l2_sum = 0
    for wm in model.model
        l2_sum += sum(abs2, wm.weight)
    end 
    return l2_sum
end

function l2_penalty(model::FE_model)
    return sum(abs2, model.embed_1.weight) + sum(abs2, model.embed_2.weight) + sum(abs2, model.hl1.weight) + sum(abs2, model.hl2.weight)
end

function l2_penalty(model::AE_model)
    l2_sum = 0 
    for wm in model.encoder
        l2_sum += sum(abs2, wm.weight)
    end 
    for wm in model.decoder
        l2_sum += sum(abs2, wm.weight)
    end 
    return l2_sum 
end


function mse_l2(model::AE_model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.net(X), Y) + l2_penalty(model) * weight_decay
end 

function mse_l2(model::FE_model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.net(X), Y) + l2_penalty(model) * weight_decay
end 

function mse_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.model(X), Y) + l2_penalty(model) * weight_decay
end 
function crossentropy_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.Losses.logitcrossentropy(model.model(X), Y) + l2_penalty(model) * weight_decay
end 
function build(model_params)
    # picks right confiration model for given params
    if model_params["model_type"] == "linear"
        chain = gpu(Dense(model_params["insize"] , model_params["outsize"],identity))
        opt = Flux.ADAM(model_params["lr"])
        lossf = crossentropy_l2
        model = logistic_regression(chain, opt, lossf)
    elseif model_params["model_type"] == "dnn"
        chain = gpu(Chain(Dense(model_params["insize"] , model_params["hl_size"], relu),
        Dense(model_params["hl_size"] , model_params["hl_size"], relu),
        Dense(model_params["hl_size"] , model_params["outsize"], identity)))
        opt = Flux.ADAM(model_params["lr"])
        lossf = crossentropy_l2
        model = dnn(chain, opt, lossf)
    elseif model_params["model_type"] == "assoc_FE"
        FE = FE_model(model_params)
        data_FE = DataFE("fe_data", model_params["fe_data"], collect(1:model_params["nsamples"]),collect(1:model_params["ngenes"]) )
        clf_chain = gpu(Flux.Chain(FE.embed_1, 
        Flux.Dense(model_params["emb_size_1"], model_params["clf_hl_size"], relu), 
        Flux.Dense(model_params["clf_hl_size"],model_params["clf_hl_size"], relu), 
        Flux.Dense(model_params["clf_hl_size"],model_params["nclasses"], identity)))
        clf_opt = Flux.ADAM(model_params["lr"])
        clf_lossf = crossentropy_l2
        clf = dnn(clf_chain, clf_opt, clf_lossf)
        model = assoc_FE(FE, data_FE , clf)
    elseif model_params["model_type"] == "assoc_ae"
        AE = AE_model(model_params)
        clf_chain = gpu(Flux.Chain(
        AE.encoder...,
        Flux.Dense(model_params["dim_redux"], model_params["clf_hl_size"], relu), 
        Flux.Dense(model_params["clf_hl_size"], model_params["clf_hl_size"], relu), 
        Flux.Dense(model_params["clf_hl_size"], model_params["nclasses"], identity)))
        clf_opt = Flux.ADAM(model_params["lr"])
        clf_lossf = crossentropy_l2
        clf = dnn(clf_chain, clf_opt, clf_lossf)
        model = assoc_AE(AE, clf)
    end 
   
    return model 
end

function train!(model::AE_model, fold;nepochs = 500, batchsize = 500, wd = 1e-6)
    ## Vanilla Auto-Encoder training function 
    train_x = fold["train_x"]';
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    for iter in 1:nepochs
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_ = gpu(train_x[:,mb_ids])
        lossval = model.lossf(model, X_, X_, weight_decay = 1e-6)
        ps = Flux.params(model.net)
        gs = gradient(ps) do
            model.lossf(model, X_, X_, weight_decay = 1e-6)
        end
        Flux.update!(model.opt, ps, gs)
        # println(my_cor(vec(X_), vec(model.net(X_))))
    end
end 
function train!(model::assoc_AE, fold; nepochs = 1000, batchsize=500, wd = 1e-6)
    ## Associative Auto-Encoder + Classifier NN model training function 
    ## Vanilla Auto-Encoder training function 
    train_x = fold["train_x"]';
    train_y = fold["train_y"]';
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    learning_curve = []
    for iter in ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_, Y_ = gpu(train_x[:,mb_ids]), gpu(train_y[:,mb_ids])
        ## gradient Auto-Encoder 
        ae_loss = model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
        ps = Flux.params(model.ae.net)
        gs = gradient(ps) do
            model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
        end
        Flux.update!(model.ae.opt, ps, gs)
        ae_cor = my_cor(vec(X_), vec(model.ae.net(X_)))
        ## gradient Classifier
        clf_loss = model.clf.lossf(model.clf, X_, Y_, weight_decay = wd)
        ps = Flux.params(model.clf.model)
        gs = gradient(ps) do
            model.clf.lossf(model.clf, X_, Y_, weight_decay = wd)
        end
        Flux.update!(model.clf.opt, ps, gs)
        clf_acc = accuracy(Y_, model.clf.model(X_))
        push!(learning_curve, (ae_loss, ae_cor, clf_loss, clf_acc))
        # save model (bson) every epoch if specified 
        #println("$iter\t AE-loss: $ae_loss\t AE-cor: $ae_cor\t CLF-loss: $clf_loss\t CLF-acc: $clf_acc")
    end
    return learning_curve
end 

function train!(model::assoc_FE, fold; nepochs = 1000, batchsize=500, wd = 1e-6)
    fe_x, fe_y = prep_FE(model.fe_data);
    nminibatches = Int(floor(length(fe_y) / batchsize))
    nsamples = length(tcga_prediction.rows)
    for iter in ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(fe_y)))
        X_i, Y_i = (fe_x[1][mb_ids],fe_x[2][mb_ids]), fe_y[mb_ids];
        model.fe_data
        lossval_fe = model.fe.lossf(model.fe, X_i, Y_i, weight_decay = wd)
        ps = Flux.params(model.fe.net)
        gs = gradient(ps) do 
            model.fe.lossf(model.fe, X_i, Y_i, weight_decay = wd)
        end
        Flux.update!(model.fe.opt,ps,gs)
        corr = my_cor(model.fe.net(X_i), Y_i)

        # training classes
        Yc = gpu(fold["train_y"]')
        Xc = gpu(fold["train_ids"])

        # gradient on classif
        ps = Flux.params(model.clf.model)
        grads = gradient(ps) do 
            model.clf.lossf(model.clf, Xc, Yc)
        end 
        Flux.update!(model.clf.opt, ps, grads)
        lossval_clf = model.clf.lossf(model.clf, Xc, Yc)
        acc = accuracy(Yc, model.clf.model(Xc))

        #println("$iter, FE-loss: $lossval_fe, FE-acc: $corr, CLF-loss: $lossval_clf, CLF-acc: $acc")
    end
end 


function train!(model::dnn, fold; nepochs = 1000, batchsize=500, wd = 1e-6)
    train_x = fold["train_x"]';
    train_y = fold["train_y"]';
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    lossf = model.lossf
    for iter in ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_, Y_ = gpu(train_x[:,mb_ids]), gpu(train_y[:,mb_ids])
        
        loss_val = lossf(model, X_, Y_, weight_decay = wd)
        ps = Flux.params(model.model)
        gs = gradient(ps) do
            lossf(model,X_, Y_, weight_decay = wd)
        end
        Flux.update!(model.opt, ps, gs)
        # println(accuracy(gpu(train_y), model.model(gpu(train_x))))
    end 
    return accuracy(gpu(train_y), model.model(gpu(train_x)))
end 

function train!(model::logistic_regression, fold; batchsize = 500, nepochs = 1000, wd = 1e-6)
    train_x = gpu(fold["train_x"]');
    train_y = gpu(fold["train_y"]');
    lossf = model.lossf
    for e in ProgressBar(1:nepochs) 
        loss_val = lossf(model, train_x, train_y, weight_decay = wd)
        ps = Flux.params(model.model)
        gs = gradient(ps) do
            lossf(model,train_x, train_y, weight_decay = wd)
        end
        Flux.update!(model.opt, ps, gs)
        #println(accuracy(model.model, train_x, train_y))
    end 
    return accuracy(train_y, model.model(train_x))
end 
function test(model::assoc_AE, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.clf.model(test_x)) 
end 

function test(model::logistic_regression, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.model(test_x)) 
end 
function test(model::dnn, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.model(test_x)) 
end
function test(model::assoc_FE, fold)
    test_x = gpu(fold["test_ids"]);
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.clf.model(test_x)) 
end
function accuracy(true_labs, pred_labs)
    n = size(true_labs)[2]
    preds = pred_labs .== maximum(pred_labs, dims = 1)
    acc = true_labs .& preds
    pct = sum(acc) / n
    return pct
end 
 
function validate(model_params, cancer_data::GDC_data; nfolds = 10)
    X = cancer_data.data
    targets = label_binarizer(cancer_data.targets)
    ## performs cross-validation with model on cancer data 
    folds = split_train_test(X, targets, nfolds = nfolds)
    true_labs_list, pred_labs_list = [],[]
    for (foldn, fold) in enumerate(folds)
        model = build(model_params)
        train_metrics = train!(model, fold, nepochs = model_params["nepochs"], batchsize = model_params["mb_size"], wd = model_params["wd"])
        true_labs, pred_labs = test(model, fold)
        push!(true_labs_list, true_labs)
        push!(pred_labs_list, pred_labs)
        println("train: ", train_metrics)
        println("test: ", accuracy(true_labs, pred_labs))
        # post run 
        # save model
        # save 2d embed svg
        # training curves svg, csv 
    end
    ### bootstrap results get 95% conf. interval 
    low_ci, med, upp_ci = bootstrap(accuracy, true_labs_list, pred_labs_list) 
    ### returns a dict 
    ret_dict = Dict("cv_acc_low_ci" => low_ci,
    "cv_acc_upp_ci" => upp_ci,
    "cv_acc_median" => med
    )
    model_params["cv_acc_low_ci"] = low_ci
    model_params["cv_acc_median"] = med
    model_params["cv_acc_upp_ci"] = upp_ci
    # param dict 
    return ret_dict
end 

function bootstrap(acc_function, tlabs, plabs; bootstrapn = 1000)
    nsamples = sum([size(tbl)[2] for tbl in tlabs])
    tlabsm = hcat(tlabs...);
    plabsm = hcat(plabs...);
    accs = []
    for i in 1:bootstrapn
        sample = rand(1:nsamples, nsamples);
        push!(accs, acc_function(tlabsm[:,sample], plabsm[:,sample]))
    end 
    sorted_accs = sort(accs)

    low_ci, med, upp_ci = sorted_accs[Int(round(bootstrapn * 0.025))], median(sorted_accs), sorted_accs[Int(round(bootstrapn * 0.975))]
    return low_ci, med, upp_ci
end 
