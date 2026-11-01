#!/usr/bin/env bash
# ==============================================================================
# Script de treinamento completo para algoritmos de fine-tuning (JAX via uv)
# ==============================================================================
set -euo pipefail

# ==============================================================================
# CONFIGURAÇÕES DO TREINAMENTO (Edite as variáveis abaixo conforme necessário)
# ==============================================================================
# Algoritmos a serem executados na ordem definida
# Opções disponíveis: "iql" "bc" "awac" "cql" "td3_bc" "dt"
ALGORITHMS=(
  "iql"
  "bc"
)

# Número de iterações
OFFLINE_ITERATIONS=1000000   # 1M steps offline
ONLINE_ITERATIONS=500000     # 500k steps online

# Tipos de comando para o ambiente
COMMAND_TYPES=(
  "forwardfixed"
  "direction"
)

# Datasets utilizados
DATASETS=(
  "playground/go2-flat-forward/expert-v0"
  "playground/go2-flat-forward/medium-v0"
)

# Ambiente de simulação
ENV_NAME="Go2JoystickFlatTerrain"

# Configurações de execução e W&B
EVAL_FREQ=50000              # Frequência de avaliação (50k steps)
PROJECT="CORL"
DEVICE="cuda"
SEED=0
CHECKPOINTS_BASE_DIR="checkpoints"

# Comando uv (pode ser sobrescrito via variável de ambiente UV)
UV="${UV:-uv}"

# Diretório raiz do repositório
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Caminho dos datasets do Minari (usa $REPO_ROOT/datasets se populado, senão ~/.minari/datasets)
if [[ -z "${MINARI_DATASETS_PATH:-}" ]]; then
  if [[ -d "$REPO_ROOT/datasets/playground" || -d "$REPO_ROOT/datasets/go2-flat-forward" ]]; then
    export MINARI_DATASETS_PATH="$REPO_ROOT/datasets"
  else
    export MINARI_DATASETS_PATH="${HOME}/.minari/datasets"
  fi
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "$REPO_ROOT"

# ==============================================================================
# MAPAS E UTILITÁRIOS
# ==============================================================================
get_module() {
  local algo="$1"
  case "$algo" in
    iql)    echo "algorithms.finetune.iql_jax" ;;
    bc)     echo "algorithms.finetune.bc_jax" ;;
    awac)   echo "algorithms.finetune.awac_jax" ;;
    cql)    echo "algorithms.finetune.cql_jax" ;;
    td3_bc) echo "algorithms.finetune.td3_bc_jax" ;;
    dt)     echo "algorithms.finetune.dt_jax" ;;
    *)
      echo "Erro: algoritmo '$algo' desconhecido." >&2
      exit 1
      ;;
  esac
}

get_algo_upper() {
  local algo="$1"
  case "$algo" in
    iql)    echo "IQL" ;;
    bc)     echo "BC" ;;
    awac)   echo "AWAC" ;;
    cql)    echo "CQL" ;;
    td3_bc) echo "TD3-BC" ;;
    dt)     echo "DT" ;;
    *)      echo "$algo" ;;
  esac
}

get_eval_flag() {
  local algo="$1"
  case "$algo" in
    awac) echo "--eval_frequency" ;;
    dt)   echo "--eval_every" ;;
    *)    echo "--eval_freq" ;;
  esac 
}

# ==============================================================================
# EXECUÇÃO DO PIPELINE
# ==============================================================================
total_runs=$(( ${#ALGORITHMS[@]} * ${#COMMAND_TYPES[@]} * ${#DATASETS[@]} ))
current_run=0
start_time=$(date +%s)

echo "=============================================================================="
echo "Iniciando fila de treinamentos de Fine-Tuning (usando uv)"
echo "Algoritmos:       ${ALGORITHMS[*]}"
echo "Command Types:    ${COMMAND_TYPES[*]}"
echo "Datasets:         ${DATASETS[*]}"
echo "Env:              ${ENV_NAME}"
echo "Offline steps:    ${OFFLINE_ITERATIONS}"
echo "Online steps:     ${ONLINE_ITERATIONS}"
echo "Eval frequency:   ${EVAL_FREQ}"
echo "Total de runs:    ${total_runs}"
echo "Comando:          ${UV} run -m <module>"
echo "=============================================================================="

for algo in "${ALGORITHMS[@]}"; do
  module="$(get_module "$algo")"
  algo_tag="$(get_algo_upper "$algo")"

  # Parâmetros de CLI específicos por algoritmo
  env_flag="--env"
  if [[ "$algo" == "dt" ]]; then
    env_flag="--env_name"
  fi
  eval_flag="$(get_eval_flag "$algo")"

  for cmd in "${COMMAND_TYPES[@]}"; do
    for ds in "${DATASETS[@]}"; do
      current_run=$((current_run + 1))
      ds_clean="$(echo "$ds" | tr '/' '_')"
      ckpt_path="${CHECKPOINTS_BASE_DIR}/${algo_tag}"
      group_name="${algo_tag}-Finetune"
      run_name="${algo_tag}-${ENV_NAME}-${cmd}-${ds_clean}"

      echo ""
      echo "------------------------------------------------------------------------------"
      echo "[$current_run/$total_runs] Executando: $algo_tag | Cmd: $cmd | Dataset: $ds"
      echo "Run name: $run_name"
      echo "Checkpoints dir: $ckpt_path"
      echo "Eval flag: $eval_flag=$EVAL_FREQ"
      echo "------------------------------------------------------------------------------"

      set +e
      "$UV" run -m "$module" \
        "$env_flag=$ENV_NAME" \
        "--dataset_id=$ds" \
        "--command_type=$cmd" \
        "--checkpoints_path=$ckpt_path" \
        "$eval_flag=$EVAL_FREQ" \
        "--project=$PROJECT" \
        "--group=$group_name" \
        "--name=$run_name" \
        "--online_iterations=$ONLINE_ITERATIONS" \
        "--offline_iterations=$OFFLINE_ITERATIONS" \
        "--device=$DEVICE" \
        "--seed=$SEED"
      exit_code=$?
      set -e

      if [[ $exit_code -ne 0 ]]; then
        echo "❌ ERRO: A execução de $run_name falhou com código $exit_code." >&2
      else
        echo "✅ SUCESSO: $run_name concluído."
      fi
    done
  done
done

elapsed=$(( $(date +%s) - start_time ))
echo ""
echo "=============================================================================="
echo "Treinamentos finalizados em $((elapsed / 60)) min e $((elapsed % 60)) s."
echo "=============================================================================="
