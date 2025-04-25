import subprocess
from datetime import datetime, timedelta
import os
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

config_files = [
    "best_models_R2/adiabatic_large_fields.yaml",
    "best_models_R2/adiabatic_large_scalars.yaml",
    "best_models_R2/adiabatic_medium_fields.yaml",
    "best_models_R2/adiabatic_medium_scalars.yaml",
    "best_models_R2/adiabatic_small_fields.yaml",
    "best_models_R2/adiabatic_small_scalars.yaml",
    "best_models_R2/Tw_large_fields.yaml",
    "best_models_R2/Tw_large_scalars.yaml",
    "best_models_R2/Tw_medium_fields.yaml",
    "best_models_R2/Tw_medium_scalars.yaml",
    "best_models_R2/Tw_small_fields.yaml",
    "best_models_R2/Tw_small_scalars.yaml",
    "best_models_NRMSE/adiabatic_large_fields.yaml",
    "best_models_NRMSE/adiabatic_large_scalars.yaml",
    "best_models_NRMSE/adiabatic_medium_fields.yaml",
    "best_models_NRMSE/adiabatic_medium_scalars.yaml",
    "best_models_NRMSE/adiabatic_small_fields.yaml",
    "best_models_NRMSE/adiabatic_small_scalars.yaml",
    "best_models_NRMSE/Tw_large_fields.yaml",
    "best_models_NRMSE/Tw_large_scalars.yaml",
    "best_models_NRMSE/Tw_medium_fields.yaml",
    "best_models_NRMSE/Tw_medium_scalars.yaml",
    "best_models_NRMSE/Tw_small_fields.yaml",
    "best_models_NRMSE/Tw_small_scalars.yaml",
]

log_file = "retrain_log.txt"

# número de processos em paralelo (ajuste conforme sua máquina)
MAX_WORKERS = min(2, os.cpu_count())  # ou 6~12 se tua máquina for parruda

def rodar_config(config_path):
    config_name = os.path.basename(config_path)
    start_dt = datetime.now()
    start_time = start_dt.timestamp()
    
    print(f"🚀 Iniciando: {config_name} às {start_dt.strftime('%H:%M:%S')}")
    
    try:
        subprocess.run(["frog", "nn", "train", config_path], check=True)
        status = "✔️ Sucesso"
    except subprocess.CalledProcessError as e:
        status = f"❌ Erro: {e}"

    end_time = datetime.now().timestamp()
    elapsed = timedelta(seconds=int(end_time - start_time))

    return {
        "config": config_name,
        "start": start_dt,
        "end": datetime.now(),
        "elapsed": elapsed,
        "status": status
    }

# salvar log de início
with open(log_file, "w") as log:
    log.write("Log de Execução dos Treinamentos\n")
    log.write(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write("=" * 60 + "\n\n")



with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(rodar_config, config) for config in config_files]

    for future in as_completed(futures):
        result = future.result()
        print(f"{result['status']} {result['config']} em {result['elapsed']}")
        print("-" * 60)

        with open(log_file, "a") as log:
            log.write(f"{result['config']}\n")
            log.write(f"  Início:  {result['start'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"  Fim:     {result['end'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"  Duração: {result['elapsed']}\n")
            log.write(f"  Status:  {result['status']}\n\n")

print("✅ Todos os treinamentos foram executados.")
