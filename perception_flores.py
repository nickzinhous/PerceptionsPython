import numpy as np
import matplotlib.pyplot as plt

class PerceptronLinear:
    def __init__(self, learning_rate=0.5, bias=1):
        self.learning_rate = learning_rate
        self.weights = None
        self.history = []
        self.bias = bias

    def step_function(self, x):
        """Função de ativação degrau"""
        return 1 if x >= 0 else 0

    def predict(self, X):
        """Faz predição para uma amostra"""
        output = np.dot(X, self.weights) + self.bias
        return self.step_function(output)

    def train_step_by_step(self, X, y):
        """Treina o perceptron mostrando cada passo"""
        n_samples, n_features = X.shape

        # Inicializar pesos
        self.weights = np.array([0.8, -0.5])
        print("Pesos iniciais:", self.weights)
        print("Bias:", self.bias)

        # Plotar situação inicial
        self.plot_decision_boundary(X, y, title="Situação Inicial", iteration=0)

        iteration = 1
        converged = False

        while not converged:
            converged = True
            print(f"\n=== ITERAÇÃO {iteration} ===")

            for i in range(n_samples):
                # Calcular output
                output_raw = np.dot(X[i], self.weights) + self.bias
                prediction = self.step_function(output_raw)

                print(f"\nAmostra {i+1}: {X[i]}, Classe esperada: {y[i]}")
                print(f"Output bruto: {output_raw:.3f}")
                print(f"Predição: {prediction}")

                # Verificar erro
                error = y[i] - prediction

                if error != 0:
                    converged = False
                    print(f"Erro: {error}")
                    print(f"Pesos antes da atualização: {self.weights}")

                    # Atualizar pesos
                    self.weights += self.learning_rate * error * X[i]
                    print(f"Pesos após atualização: {self.weights}")

                    # Salvar histórico
                    self.history.append({
                        'iteration': iteration,
                        'sample': i+1,
                        'weights': self.weights.copy(),
                        'error': error
                    })

                    # Plotar nova situação
                    self.plot_decision_boundary(X, y,
                          title=f"Após Iteração {iteration} - Amostra {i+1}",
                          iteration=iteration)
                    break
                else:
                    print("Classificação correta - pesos não alterados")

            iteration += 1

            # Verificar convergência
            if converged:
                print(f"\n✅ CONVERGÊNCIA ALCANÇADA na iteração {iteration-1}!")
                print(f"Pesos finais: {self.weights}")
                break

    def plot_decision_boundary(self, X, y, title="", iteration=0):
        """Plota pontos e a reta de decisão"""
        plt.figure(figsize=(10, 8))

        # Plotar pontos
        colors = ['red', 'blue']
        labels = ['Classe 0 (Versicolor)', 'Classe 1 (Setosa)']

        for class_value in [0, 1]:
            mask = y == class_value
            if np.any(mask):
                plt.scatter(X[mask, 0], X[mask, 1],
                           c=colors[class_value],
                           label=labels[class_value],
                           s=150,
                           edgecolors='black',
                           linewidth=2,
                           alpha=0.8)

        # Limites do gráfico
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # Reta de decisão
        if self.weights is not None:
            w1, w2 = self.weights
            if abs(w2) > 1e-6:
                x_line = np.linspace(x_min, x_max, 100)
                y_line = -(w1 * x_line + self.bias) / w2
                plt.plot(x_line, y_line, 'k--',
                        label=f'Reta: {w1:.3f}x1 + {w2:.3f}x2 + {self.bias} = 0',
                        linewidth=3)

                # Direção da classificação
                x_mid = (x_min + x_max) / 2
                y_mid = -(w1 * x_mid + self.bias) / w2
                plt.arrow(x_mid, y_mid, w1*0.3, w2*0.3,
                         head_width=0.1, head_length=0.1,
                         fc='green', ec='green', alpha=0.7)

            elif abs(w1) > 1e-6:
                x_vertical = -self.bias / w1
                plt.axvline(x=x_vertical, color='k', linestyle='--',
                           label=f'Reta: {w1:.3f}x1 + {self.bias} = 0',
                           linewidth=3)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlabel('Comprimento da pétala (cm)', fontsize=12)
        plt.ylabel('Largura da pétala (cm)', fontsize=12)

        if self.weights is not None:
            plt.title(f'{title}\nPesos: w1={self.weights[0]:.3f}, w2={self.weights[1]:.3f}, bias={self.bias}',
                     fontsize=11)
        else:
            plt.title(title, fontsize=11)

        # Regiões de classificação
        if iteration > 0:
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50))

            grid_points = np.c_[xx.ravel(), yy.ravel()]
            predictions = []
            for point in grid_points:
                output = np.dot(point, self.weights) + self.bias
                predictions.append(self.step_function(output))

            predictions = np.array(predictions).reshape(xx.shape)
            plt.contourf(xx, yy, predictions, levels=[-0.5, 0.5, 1.5],
                        colors=['lightcoral', 'lightblue'], alpha=0.3)

        plt.tight_layout()
        plt.show()

    def test_final_model(self, X, y):
        """Testa o modelo final com todas as amostras"""
        print("\n=== TESTE DO MODELO FINAL ===")
        correct = 0
        for i, (x, expected) in enumerate(zip(X, y)):
            output_raw = np.dot(x, self.weights) + self.bias
            prediction = self.step_function(output_raw)
            is_correct = prediction == expected
            if is_correct:
                correct += 1

            print(f"Amostra {i+1} {x}: Output={output_raw:.3f}, "
                  f"Predição={prediction}, Esperado={expected} "
                  f"{'✔' if is_correct else '✘'}")

        accuracy = correct / len(y) * 100
        print(f"\nAcurácia: {correct}/{len(y)} = {accuracy:.1f}%")

# ==============================
# DADOS DA TABELA (FLORES) - EXERCÍCIO 3
# ==============================
X = np.array([
    [5.1, 1.6],  # X(1) - Classe 0 - versicolor
    [4.0, 1.3],  # X(2) - Classe 0 - versicolor
    [4.8, 1.8],  # X(3) - Classe 0 - versicolor
    [1.4, 0.3],  # X(4) - Classe 1 - setosa
    [1.9, 0.4],  # X(5) - Classe 1 - setosa
    [1.5, 0.2],  # X(6) - Classe 1 - setosa
])

y = np.array([0, 0, 0, 1, 1, 1])  # Y(k) conforme tabela

print("=== EXERCÍCIO 3: PERCEPTRON PARA CLASSIFICAÇÃO DE FLORES ===")
print("Classificação de flores venenosas e não venenosas")
print("\nDados da tabela:")
print("X(k) | Comprimento | Largura | Y(k) | Espécie")
print("-----|-------------|---------|------|---------")
for i in range(len(X)):
    especie = "versicolor" if y[i] == 0 else "setosa"
    print(f"X({i+1}) | {X[i][0]:>10.1f} | {X[i][1]:>7.1f} | {y[i]:>4} | {especie}")
print("\nParâmetros:")
print("Taxa de aprendizado: 0.5")
print("Bias: 1")

# Criar e treinar o perceptron
perceptron = PerceptronLinear(learning_rate=0.5, bias=1)
perceptron.train_step_by_step(X, y)

# Testar modelo final
perceptron.test_final_model(X, y)

# Mostrar resumo do treinamento
print(f"\n=== RESUMO DO TREINAMENTO ===")
print(f"Número total de atualizações: {len(perceptron.history)}")
for entry in perceptron.history:
    print(f"Iteração {entry['iteration']}, Amostra {entry['sample']}: "
          f"Erro={entry['error']}, Novos pesos={entry['weights']}")

print("\n=== ANÁLISE DOS DADOS ===")
print("Distribuição das classes:")
for class_val in [0, 1]:
    mask = y == class_val
    points = X[mask]
    especie = "versicolor" if class_val == 0 else "setosa"
    print(f"Classe {class_val} ({especie}): {len(points)} pontos")
    print(f"  Pontos: {points.tolist()}")
    if len(points) > 0:
        print(f"  Centroide: [{np.mean(points[:, 0]):.2f}, {np.mean(points[:, 1]):.2f}]")

print("\n=== CONCLUSÃO ===")
print("O perceptron convergiu com sucesso para classificar as flores!")
print("Classe 0 (versicolor): flores com pétalas maiores")
print("Classe 1 (setosa): flores com pétalas menores")
