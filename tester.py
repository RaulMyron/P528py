import pandas as pd
from p528 import P528

def executa_programa_final2_df(distancias, alturas_terminal1, alturas_terminal2, frequencia, polarizacao, porcentagem_tempo):
    resultados = []

    i = 0

    for distancia in distancias:
        linha_resultados = {'D (km)': distancia}

        for altura_terminal1 in alturas_terminal1:
            for altura_terminal2 in alturas_terminal2:
                # Create an instance of P528 and compute the loss
                
                print(f'instancia {i}:', distancia, altura_terminal1, altura_terminal2, frequencia, polarizacao, porcentagem_tempo)

                p528_instance = P528(distancia, altura_terminal1, altura_terminal2, frequencia, polarizacao, porcentagem_tempo)
                print(p528_instance)
                perda = float(p528_instance.A__db)

                # Add the result to the corresponding row for height and distance
                coluna = f"h1={altura_terminal1}_h2={altura_terminal2}"
                linha_resultados[coluna] = perda
        
                i+=1
        # Add the results for the current distance to the final list
        resultados.append(linha_resultados)
        

    # Create the DataFrame from the results
    df_resultados = pd.DataFrame(resultados)
    
    return df_resultados

# Define the distances, heights, and other parameters
distancias = range(0, 501)  # Distances from 0 to 500 km
alturas_terminal1 = [1.5, 15, 30, 60, 1000, 10000, 20000]
alturas_terminal2 = [1000, 1000, 10000, 20000]
frequencia = 1200  # Example frequency provided
polarizacao = 0
porcentagem_tempo = 1

# Execute the function and generate the DataFrame
df = executa_programa_final2_df(distancias, alturas_terminal1, alturas_terminal2, frequencia, polarizacao, porcentagem_tempo)

# Display the resulting DataFrame
print(df)
