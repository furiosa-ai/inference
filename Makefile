# end-to-end evaluation
.PHONY: all
all: llama2 stablediffusion

.PHONY: llama2
llama2:
	-bash scripts/build_llama2-70b_env.sh
	-bash scripts/eval_llama2-70b.sh

.PHONY: stablediffusion
stablediffusion:
	-bash scripts/build_stablediffusion_env.sh
	-bash scripts/eval_stablediffusion.sh
