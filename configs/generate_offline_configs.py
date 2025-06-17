#!/usr/bin/env python3
import yaml
import argparse
from pathlib import Path
import re

# Script deve ficar em configs/
ROOT_DIR = Path(__file__).parent

def extract_prefix(folder_name: str) -> str:
    lower = folder_name.lower()
    if lower.startswith("go1"): return "go1"
    if lower.startswith("g1"):  return "g1"
    if lower.startswith("h1"):  return "h1"
    return lower.replace("/", "-").replace("_", "-")

# Converte CamelCase para snake_case
def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

# Parâmetros padrão por algoritmo
def get_algorithm_defaults(algorithm: str) -> dict:
    defaults = {
        "awac": {
            "project": "CORL",
            "awac_lambda": 0.1,
            "hidden_dim": 256,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 10000000,
            "eval_frequency": 1000,
            "n_test_episodes": 10,
            "num_train_ops": 1000000,
            "seed": 42,
            "test_seed": 69,
            "device": "cuda",
            "deterministic_torch": False,
            "checkpoints_path": "AWAC",
        },
        "iql": {
            "project": "CORL",
            "discount": 0.99,
            "tau": 0.005,
            "beta": 3.0,
            "iql_tau": 0.8,
            "iql_deterministic": False,
            "max_timesteps": 1000000,
            "buffer_size": 10000000,
            "batch_size": 256,
            "normalize": True,
            "normalize_reward": False,
            "vf_lr": 3e-4,
            "qf_lr": 3e-4,
            "actor_lr": 3e-4,
            "actor_dropout": 0.1,
            "eval_freq": 5000,
            "n_episodes": 10,
            "seed": 0,
            "device": "cuda",
            "load_model": "",
            "checkpoints_path": None,
        }
    }
    if algorithm not in defaults:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")
    return defaults[algorithm].copy()

# Padrões de dataset_id e group por algoritmo
PATTERNS = {
    "awac": {
        "dataset_pattern": "playground/{env}-{level}-v0",
        "group_pattern": "awac-{prefix}-{level}-v0"
    },
    "iql": {
        "dataset_pattern": "{env}-{level}",
        "group_pattern": "iql-{env}-{level}-multiseed-v0"
    }
}


def generate_config(algorithm: str, folder_name: str, env_name: str, level: str):
    # Obtém defaults e patterns
    config = get_algorithm_defaults(algorithm)
    patterns = PATTERNS[algorithm]

    # Campos dinâmicos
    prefix = extract_prefix(folder_name)
    config["env"] = env_name
    # Define dataset_id para todos os algoritmos
    config["dataset_id"] = patterns["dataset_pattern"].format(env=env_name, level=level)
    config["group"] = patterns["group_pattern"].format(env=env_name, prefix=prefix, level=level)

    # Monta path de saída: configs/offline/{algorithm}/{folder_name}/{snake}_{level}.yaml
    out_dir = ROOT_DIR / "offline" / algorithm / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{camel_to_snake(env_name)}_{level}.yaml"
    out_file = out_dir / file_name

    # Salva YAML
    with open(out_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Gerado: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera config offline baseado em parâmetros --algorithm, --folder_name, --env, --level"
    )
    parser.add_argument(
        "--algorithm", required=True, choices=list(PATTERNS.keys()),
        help="Algoritmo: awac ou iql"
    )
    parser.add_argument(
        "--folder_name", required=True,
        help="Pasta do ambiente: go1, g1, h1, etc."
    )
    parser.add_argument(
        "--env", required=True,
        help="Nome do ambiente exato para o campo 'env'"
    )
    parser.add_argument(
        "--level", required=True, choices=["random", "medium", "expert"],
        help="Nível do dataset"
    )

    args = parser.parse_args()
    generate_config(args.algorithm, args.folder_name, args.env, args.level)


if __name__ == "__main__":
    main()
