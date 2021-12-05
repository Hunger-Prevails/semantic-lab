export CUDA_VISIBLE_DEVICES=2

python main.py \
	-pretrain COCO \
	-shuffle \
	-data_name smile_view \
	-backbone resnet50 \
	-head deeplabv3 \
	-save_path /home/yinglun.liu/checkpoints \
	-suffix debug \
	-adapter Polynom \
	-criterion CrossEntropy \
	-n_cudas 1 \
	-learn_rate 2e-3
