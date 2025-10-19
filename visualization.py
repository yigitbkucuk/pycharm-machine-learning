import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Gerçek Etiketler')
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.savefig(save_path)
    plt.close()


def plot_sample_predictions(X_test, y_test, y_pred, n_samples=10, save_path='results/sample_predictions.png'):
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        color = 'green' if y_pred[idx] == y_test[idx] else 'red'
        plt.title(f'Gerçek: {y_test[idx]}\nTahmin: {y_pred[idx]}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_k_analysis(k_values, accuracies, save_path='results/k_value_analysis.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    best_idx = np.argmax(accuracies)
    plt.scatter(k_values[best_idx], accuracies[best_idx], color='red', s=100, zorder=5)
    plt.xlabel('k Değeri')
    plt.ylabel('Accuracy')
    plt.title('K Değerinin Accuracy Üzerindeki Etkisi')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_distance_comparison(k_values, l1_accuracies, l2_accuracies, save_path='results/distance_comparison.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, l1_accuracies, marker='o', label='L1 (Manhattan)', linewidth=2)
    plt.plot(k_values, l2_accuracies, marker='s', label='L2 (Euclidean)', linewidth=2)
    plt.xlabel('k Değeri')
    plt.ylabel('Accuracy')
    plt.title('L1 vs L2 Mesafe Metrikleri Karşılaştırması')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def create_comparison_table(k_values, l1_accuracies, l2_accuracies, save_path='results/comparison_table.png'):
    differences = [abs(l1 - l2) for l1, l2 in zip(l1_accuracies, l2_accuracies)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['K Değeri', 'L1 Accuracy', 'L2 Accuracy', 'Fark'])
    for k, l1, l2, diff in zip(k_values, l1_accuracies, l2_accuracies, differences):
        table_data.append([k, f'{l1:.4f}', f'{l2:.4f}', f'{diff:.4f}'])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('L1 ve L2 Mesafe Metrikleri Karşılaştırma Tablosu')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()