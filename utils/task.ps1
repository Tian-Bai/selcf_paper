cd "D:\Github\selcf_paper\utils"

$args0 = "-i", "1", "-c", "True", "mlp", "hidden", "-r", "1,2,1", "layers:2" # for testing

$args1 = @("-n", "1000", "-c", "True", "mlp", "hidden", "-r", "1,21,1", "layers:2")
$args2 = @("-n", "1000", "-c", "False", "mlp", "hidden", "-r", "1,21,1", "layers:2")

$args3 = @("-n", "1000", "-c", "True", "rf", "n_estim", "-r", "1,51,2", "max_features:10,max_leaf_nodes:None,max_depth:30")
$args4 = @("-n", "1000", "-c", "False", "rf", "n_estim", "-r", "1,51,2", "max_features:10,max_leaf_nodes:None,max_depth:30")

$args5 = @("-n", "1000", "-c", "True", "rf", "max_depth", "-r", "1,51,2", "max_features:10,max_leaf_nodes:None,n_estim:50")
$args6 = @("-n", "1000", "-c", "False", "rf", "max_depth", "-r", "1,51,2", "max_features:10,max_leaf_nodes:None,n_estim:50")

# python gendata-model.py @args1
# python gendata-model.py @args2
# python gendata-model.py @args3
# done

# python gendata-model.py @args4
python gendata-model.py @args5
python gendata-model.py @args6