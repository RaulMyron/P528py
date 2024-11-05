# flake8: noqa
import pandas as pd
import itertools
import time
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor
from scalar.p528 import *

# Função para medir uso de CPU e memória


def get_system_metrics():
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=1)  # Mede uso de CPU durante 1 segundo
    return mem_info.rss / (1024 * 1024), cpu_percent  # Retorna em MB

# Função para executar o P528


def run_p528(params, terminal_1, terminal_2, tropo, path, los_params):
    d, h1, h2, f, T_pol, p = params
    try:
        
        result = P528_Ex(d, h1, h2, f, T_pol, p, result,
                           terminal_1, terminal_2, tropo, path, los_params)

        # Retorna os resultados
        return {
            "d__km": d,
            "h_1__meter": h1,
            "h_2__meter": h2,
            "f__mhz": f,
            "T_pol": T_pol,
            "p": p,
            "Perda_total_dB": result.A__db
        }
    except Exception as e:
        print(f"Erro ao processar {params}: {e}")
        return None
    finally:
        gc.collect()  # Chama o coletor de lixo para liberar memória
        
    
def main():
    
    # Definição dos parâmetros
    d__km = range(0, 1001)  # Distâncias de 0 a 1000 km
    h_1__meter = [1.5, 15, 30, 60, 1000, 10000, 20000]
    h_2__meter = [1000, 10000, 20000]
    f__mhz = [100, 125, 300, 600, 1200, 2400, 5100, 9400, 15500, 30000]
    T_pol = 0  # Polarização horizontal
    p = [1, 5, 10, 50, 95]  # Percentuais de tempo

    # Criar uma lista de todas as combinações de parâmetros 
    combinations = list(itertools.product(d__km, h_1__meter, h_2__meter, f__mhz, [T_pol], p))
    
    total_combinations = len(combinations)  # Total de combinações
    start_time = time.time()
    initial_memory, initial_cpu = get_system_metrics()
    results = []

    # Tamanho máximo para o lote de combinações
    BATCH_SIZE = 100000
    
    terminal_1 = Terminal()
    terminal_2 = Terminal()
    tropo = TroposcatterParams()
    path = Path()
    los_params = LineOfSightParams()
    result = Result()

    # Usar ProcessPoolExecutor para executar em paralelo
    with ProcessPoolExecutor(max_workers=21) as executor:
        # Especifica 10 núcleos
        # Divida as combinações em lotes
        
        for batch in range(0, total_combinations, BATCH_SIZE):
            current_batch = combinations[batch:batch + BATCH_SIZE]
            for i, result in enumerate(executor.map(run_p528, current_batch)):
                lambda params: run_p528(params + (terminal_1, terminal_2, tropo, path, los_params))
                if result is not None:
                    results.append(result)
                # Imprime quantas combinações faltam
                remaining = total_combinations - (batch + i + 1)
                print(f"Faltam {remaining} combinações.")
                
                result.clear(); terminal_1.clear(); terminal_2.clear(); tropo.clear(); path.clear(); los_params.clear()

    # Medir tempo final e coletar métricas finais de sistema
    end_time = time.time()
    final_memory, final_cpu = get_system_metrics()

    # Calcular resultados de desempenho
    elapsed_time = end_time - start_time
    memory_used = final_memory - initial_memory
    cpu_usage = final_cpu - initial_cpu

    # Criar um DataFrame com os resultados
    df = pd.DataFrame(results)

    # Salvar os resultados em um arquivo CSV
    df.to_csv("p528_results_Paraleliza.csv", index=False)
    # Exibir resultados de desempenho
    print("Tempo total de execução:", elapsed_time, "segundos")
    print("Memória utilizada:", memory_used, "MB")
    print("Uso de CPU:", cpu_usage, "%")
    print("Execução concluída. Resultados salvos em 'p528_results_Paraleliza.csv'.")


if __name__ == '__main__':
    #print('hello')
    main()
    
