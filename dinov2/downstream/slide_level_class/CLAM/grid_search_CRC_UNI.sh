lr=(1e-4 5e-5 1e-5)
dropout=(0.25 0.5)
reg=(1e-4 1e-5)
loss=(ce focal)
alpha=(0.25 0.4 0.5)
gamma=(1 2 3)
stopping=(loss error auc)
weighted_sample=(True False)
CUDA_id=0

# Loop through each combination of hyperparameters
for p1 in "${lr[@]}"; do
  for p2 in "${dropout[@]}"; do
    for p3 in "${reg[@]}"; do
      for p4 in "${loss[@]}"; do
        for p7 in "${stopping[@]}"; do
          for p8 in "${weighted_sample[@]}"; do
          # If p4 equals "focal", loop through alpha and gamma combinations
            if [[ "$p4" == "focal" ]]; then
              for p5 in "${alpha[@]}"; do
                for p6 in "${gamma[@]}"; do
                  # Run model with alpha and gamma for focal loss
                  CUDA_id=$((CUDA_id % 8))
                  CUDA_VISIBLE_DEVICES=$CUDA_id python main.py --drop_out $p2 --early_stopping \
                    --lr $p1 --k 1 --weighted_sample $p8 --bag_loss focal --alpha $p5 --gamma $p6 --task msi_vs_mss --reg $p3 \
                    --model_type clam_abmil --model_size big --feats_dir feats_UNI --data_root_dir /mnt/nfs03-R6/TCGA/crc_msi_cls/ \
                    --exp_code CRC_UNI --embed_dim 1024 --stop_epoch 20 --patience 15 --log_data --no_inst_cluster --stopping_criterion $p7 \
                    --results_dir ./results/CRC --csv_path crc_msi_vs_mss_new.csv --para_group "$p1"+"$p2"+"$p3"+"$p4"+"$p5"+"$p6"+"$p7"+"$p8"
                  CUDA_id=$((CUDA_id+1))
                done
              done
            else
              # Run model without alpha and gamma for ce loss
              CUDA_id=$((CUDA_id % 8))
              CUDA_VISIBLE_DEVICES=$CUDA_id python main.py --drop_out $p2 --early_stopping \
                --lr $p1 --k 1 --weighted_sample $p8  --bag_loss ce --task msi_vs_mss --reg $p3 \
                --model_type clam_abmil --model_size big --feats_dir feats_UNI --data_root_dir /mnt/nfs03-R6/TCGA/crc_msi_cls/ \
                --exp_code CRC_UNI --embed_dim 1024 --stop_epoch 20 --patience 15 --log_data --no_inst_cluster --stopping_criterion $p7 \
                --results_dir ./results/CRC --csv_path crc_msi_vs_mss_new.csv --para_group "$p1"+"$p2"+"$p3"+"$p4"+"$p7"+"$p8" 
              CUDA_id=$((CUDA_id+1))
            fi
          done
        done
      done
    done
  done
done

