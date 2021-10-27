export CUDA_VISIBLE_DEVICES=1

python main.py \
	-pretrain \
	-shuffle \
	-data_name smile_view \
	-backbone resnet50 \
	-head deeplabv3 \
	-save_path /home/yinglun.liu/checkpoints \
	-suffix debug \
	-criterion CrossEntropy \
	-n_cudas 1
