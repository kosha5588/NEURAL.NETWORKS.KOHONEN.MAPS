import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import sys

def load_and_validate(learn_path, pca_path):
    try:
        learn_data = pd.read_csv(learn_path, header=None, sep=r'\s+')  
        X = learn_data.values
        
        pca_data = pd.read_csv(pca_path, header=None, sep=r'\s+')  
        Z = pca_data.values
        
        if X.shape[0] != Z.shape[0]:
            raise ValueError(f"Несоответствие объектов: {X.shape[0]} vs {Z.shape[0]}")
            
        return X, Z
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

def main():
    X, Z = load_and_validate('Learning_data7.txt', 'PCA_data7.txt')
    
 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    som = MiniSom(
        x=2, y=2,          
        input_len=X.shape[1], 
        sigma=1.0, 
        learning_rate=0.5,
        neighborhood_function='gaussian',
        random_seed=42
    )
    som.pca_weights_init(X_scaled)
    som.train(X_scaled, 100, verbose=True)
    

    cluster_labels = np.array([som.winner(x)[0]*2 + som.winner(x)[1] for x in X_scaled])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(Z[:,0], Z[:,1], c=cluster_labels, 
                         cmap='viridis', s=80, edgecolor='k')
    plt.title('Проекция на главные компоненты', fontsize=14)
    plt.xlabel('Z1', fontsize=12)
    plt.ylabel('Z2', fontsize=12)
    plt.colorbar(scatter, label='Кластер')
    
    plt.subplot(1, 2, 2)
    clusters = {label: X[cluster_labels == label] for label in np.unique(cluster_labels)}
    for label, data in clusters.items():
        plt.plot(np.mean(data, axis=0), 
                marker='o', 
                linestyle='--',
                label=f'Кластер {label} (n={len(data)})')
    
    plt.title('Средние значения признаков', fontsize=14)
    plt.xlabel('Номер признака', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.xticks(range(X.shape[1]), [f'X{i+1}' for i in range(X.shape[1])])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300)
    plt.show()
    
    print("\nРезультаты анализа:")
    print(f"Всего объектов: {X.shape[0]}")
    print(f"Выявлено кластеров: {len(clusters)}")
    for label, data in clusters.items():
        print(f"Кластер {label+1}: {len(data)} объектов")
        print(f"Средние значения: {np.round(np.mean(data, axis=0), 2)}")
        print("-"*40)

if __name__ == "__main__":
    main()