import subprocess
import time
from datetime import datetime, timedelta
import os
import yaml
# Lista dos arquivos de configuração
config_files = [
    "gp_adiabatic_large_fields.yaml",
    "gp_adiabatic_large_scalars.yaml",
    "gp_adiabatic_medium_fields.yaml",
    "gp_adiabatic_medium_scalars.yaml",
    "gp_adiabatic_small_fields.yaml",
    "gp_adiabatic_small_scalars.yaml",
    "gp_Tw_large_fields.yaml",
    "gp_Tw_large_scalars.yaml",
    "gp_Tw_medium_fields.yaml",
    "gp_Tw_medium_scalars.yaml",
    "gp_Tw_small_fields.yaml",
    "gp_Tw_small_scalars.yaml",
]

log_file = "hpo_log.txt"

with open(log_file, "w") as log:
    log.write("Log de Execução dos Estudos de Otimização\n")
    log.write(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write("=" * 60 + "\n\n")

for idx, config in enumerate(config_files, 1):
    with open(config, 'r') as f:
        dados = yaml.safe_load(f)

    config_name = os.path.basename(config)
    start_dt = datetime.now()
    print(f"[{idx}/{len(config_files)}] Iniciando: {config_name} às {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        command = ["frog", "gp", "train", config]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante execução de {config_name}: {e}")
    
    end_time = time.time()
    end_dt = datetime.now()
    elapsed = timedelta(seconds=int(end_time - start_time))

    with open(log_file, "a") as log:
        log.write(f"{config_name}\n")
        log.write(f"  Início: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"  Fim:    {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"  Duração: {elapsed}\n\n")

    print(f"✔ Finalizado {config_name} em {elapsed}")
    print("-" * 60)

print("✅ Todos os estudos foram executados.")
