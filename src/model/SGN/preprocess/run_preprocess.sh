
datasets=(  
    # hest/bench_data/CCRCC
    # hest/bench_data/PRAD 
    # hest/kidney  
    nameeta/GBM
    # takano/xenium
    takano/visium
    # ST/andrew
    # ST/andersson
    # ST/bryan
)

for dataset in "${datasets[@]}"; do
    python generate_graph.py \
        --data_dir "/home/shared/chungym/hier_st/input/${dataset}" \
        --num_edge 5 
done

# python build_exemplar.py --data_dir /home/shared/chungym/hier_st/input/takano/xenium --distance_metric l1