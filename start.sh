exe=main.py
queue_size=1000
net_type=r50 # 输入大小为224x224的话，网络选择r50，112x112选择ir50
log_file=${net_type}.log
saved_dir=${net_type}pth
if
[ ! -d ${saved_dir} ]; then mkdir -p ${saved_dir}
fi
feat_dim=512
batch_size=64
loss_type=AM
snapshot=
python ${exe} ${saved_dir} --net_type=${net_type} --queue_size=${queue_size} --feat_dim=${feat_dim} --batch_size=${batch_size} --loss_type=${loss_type} --pretrained_model_path=${snapshot} > ${log_file} 2>&1 &
