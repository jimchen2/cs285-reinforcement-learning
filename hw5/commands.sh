python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_easy_random.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_medium_random.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_hard_random.yaml \
--dataset_dir datasets/ --log_interval 1000


python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_easy_rnd.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_medium_rnd.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_hard_rnd.yaml \
--dataset_dir datasets/ --log_interval 1000



#4.1

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_easy_cql.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_hard_cql.yaml \
--dataset_dir datasets --log_interval 1000

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_alpha0.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_alpha10.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_alpha1.yaml \
--dataset_dir datasets --log_interval 1000


python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_easy_dqn.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_dqn.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_hard_dqn.yaml \
--dataset_dir datasets --log_interval 1000







#4.2

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_easy_awac.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_awac.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_hard_awac.yaml \
--dataset_dir datasets --log_interval 1000


python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_easy_iql.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_iql.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_hard_iql.yaml \
--dataset_dir datasets --log_interval 1000




#4.3


python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_medium_rnd_1000.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_medium_rnd_5000.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_medium_rnd_10000.yaml \
--dataset_dir datasets/ --log_interval 1000
python cs285/scripts/run_hw5_explore.py \
-cfg experiments/exploration/pointmass_medium_rnd_20000.yaml \
--dataset_dir datasets/ --log_interval 1000

python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_rnd1000.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_rnd5000.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_rnd10000.yaml \
--dataset_dir datasets --log_interval 1000
python ./cs285/scripts/run_hw5_offline.py \
-cfg experiments/offline/pointmass_medium_cql_rnd20000.yaml \
--dataset_dir datasets --log_interval 1000
