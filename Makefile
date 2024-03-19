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

# verified evaluation log
.PHONY: log_all
log_all: log_llama2 log_stablediffusion

.PHONY: log_llama2
log_llama2:
	-dvc pull logs/internal/llama2-70b.dvc

.PHONY: log_stablediffusion
log_stablediffusion:
	-dvc pull logs/internal/stablediffusion.dvc