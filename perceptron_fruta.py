# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class PerceptronFrutas:
    def __init__(self, learning_rate=0.1, bias=1.0):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = bias
        self.history = []

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Modelo não treinado!")
        output = np.dot(X, self.weights) + self.bias
        return self.step_function(output)

    def train(self, X, y, max_iterations=100):
        n_samples, n_features = X.shape
        # Inicializar pesos com valores pequenos mas não zero
        self.weights = np.random.uniform(-0.1, 0.1, n_features)
        
        print(f"Pesos iniciais: {self.weights}")
        print(f"Bias inicial: {self.bias}")
        print(f"Taxa de aprendizado: {self.learning_rate}")

        # Adicionar estado inicial ao histórico
        self.history.append({
            'iteration': 0,
            'weights': self.weights.copy(),
            'bias': self.bias,
            'accuracy': 0
        })

        for iteration in range(1, max_iterations + 1):
            errors = 0
            print(f"\n=== ITERAÇÃO {iteration} ===")
            print(f"Pesos atuais: {self.weights}")
            
            for i in range(n_samples):
                output = np.dot(X[i], self.weights) + self.bias
                prediction = self.step_function(output)
                error = y[i] - prediction
                
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    errors += 1
                    print(f"  Amostra {i+1}: {X[i]} -> Erro: {error}")
                    print(f"  Novos pesos: {self.weights}")
            
            accuracy = self.calculate_accuracy(X, y)
            self.history.append({
                'iteration': iteration,
                'weights': self.weights.copy(),
                'bias': self.bias,
                'accuracy': accuracy
            })
            
            print(f"Acurácia: {accuracy:.2%}")
            
            if errors == 0:  # Convergência
                print(f"\n✅ CONVERGÊNCIA na iteração {iteration}!")
                break

        print(f"\nPesos finais: {self.weights}")
        print(f"Bias final: {self.bias}")

    def calculate_accuracy(self, X, y):
        predictions = [self.predict(x) for x in X]
        return np.mean(np.array(predictions) == y)

    def plot_training_history(self):
        iterations = [entry['iteration'] for entry in self.history]
        weights_history = np.array([entry['weights'] for entry in self.history])
        accuracies = [entry['accuracy'] for entry in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(iterations, weights_history[:, 0], 'b-o', label='Peso 1 (Peso)', linewidth=2)
        ax1.plot(iterations, weights_history[:, 1], 'r-s', label='Peso 2 (Acidez)', linewidth=2)
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Valor do Peso')
        ax1.set_title('Evolução dos Pesos')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(iterations, accuracies, 'g-o', linewidth=2)
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Acurácia')
        ax2.set_title('Evolução da Acurácia')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(10, 8))

        colors = ['orange', 'red']
        labels = ['Laranja (0)', 'Maçã (1)']

        for class_value in [0, 1]:
            mask = y == class_value
            if np.any(mask):
                plt.scatter(X[mask, 0], X[mask, 1],
                            c=colors[class_value],
                            label=labels[class_value],
                            s=150, edgecolors='black', linewidth=2, alpha=0.8)

        x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        if self.weights is not None:
            w1, w2 = self.weights
            if abs(w2) > 1e-6:
                x_line = np.linspace(x_min, x_max, 100)
                y_line = -(w1 * x_line + self.bias) / w2
                plt.plot(x_line, y_line, 'k--',
                         label=f'Fronteira: {w1:.3f}×Peso + {w2:.3f}×Acidez + {self.bias:.3f} = 0',
                         linewidth=3)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlabel('Peso (g)', fontsize=12)
        plt.ylabel('Acidez (pH)', fontsize=12)
        plt.title(f'Classificação Laranjas vs Maçãs\nPesos: [{self.weights[0]:.3f}, {self.weights[1]:.3f}], Bias: {self.bias:.3f}')
        plt.tight_layout()
        plt.show()

    def test_new_cases(self, new_cases):
        print("\n=== TESTE COM NOVOS CASOS ===")
        print("Formato: [Peso (g), Acidez (pH)]")
        
        for i, case in enumerate(new_cases, 1):
            prediction = self.predict(case)
            output_raw = np.dot(case, self.weights) + self.bias
            fruit = "Maçã" if prediction == 1 else "Laranja"
            
            print(f"Caso {i}: {case} -> Output: {output_raw:.3f} -> {fruit}")

# DADOS LINEARMENTE SEPARÁVEIS - Melhor separação
X_train = np.array([
    [200, 2.8],  # Laranja - muito pesada, muito ácida
    [180, 3.0],  # Laranja - pesada, muito ácida
    [190, 2.9],  # Laranja - pesada, muito ácida
    [90, 4.8],   # Maçã - leve, pouco ácida
    [100, 4.5],  # Maçã - leve, pouco ácida
    [110, 4.2],  # Maçã - média, pouco ácida
])
y_train = np.array([0, 0, 0, 1, 1, 1])  # 0 = Laranja, 1 = Maçã

print("=== PERCEPTRON PARA CLASSIFICAÇÃO DE FRUTAS (VERSÃO CORRIGIDA) ===")
print("Dados de treinamento (melhor separação):")
print("Laranjas (classe 0) - Mais pesadas E mais ácidas:")
for i, (x, y_val) in enumerate(zip(X_train, y_train)):
    if y_val == 0:
        print(f"  {i+1}. Peso: {x[0]}g, Acidez: {x[1]}pH (mais ácida)")
print("Maçãs (classe 1) - Mais leves E menos ácidas:")
for i, (x, y_val) in enumerate(zip(X_train, y_train)):
    if y_val == 1:
        print(f"  {i+1}. Peso: {x[0]}g, Acidez: {x[1]}pH (menos ácida)")

print("\nExplicação das características:")
print("- Laranjas: pH 2.8-3.0 (muito ácidas), Peso 180-200g (muito pesadas)")
print("- Maçãs: pH 4.2-4.8 (pouco ácidas), Peso 90-110g (leves)")

# Criar e treinar perceptron
perceptron = PerceptronFrutas(learning_rate=0.1, bias=1.0)
perceptron.train(X_train, y_train, max_iterations=50)

# Mostrar resultados dos pesos
print("\n=== RESULTADOS DOS PESOS A CADA ITERAÇÃO ===")
for entry in perceptron.history:
    print(f"Iteração {entry['iteration']:2d}: "
          f"Pesos = [{entry['weights'][0]:6.3f}, {entry['weights'][1]:6.3f}], "
          f"Bias = {entry['bias']:6.3f}, "
          f"Acurácia = {entry['accuracy']:5.1%}")

# Mostrar gráficos
perceptron.plot_training_history()
perceptron.plot_decision_boundary(X_train, y_train)

# Testar com novos casos
new_cases = [
    [195, 2.9],  # Laranja (peso alto, acidez alta)
    [95, 4.6],   # Maçã (peso baixo, acidez baixa)
    [185, 2.8],  # Laranja (peso alto, acidez alta)
    [105, 4.3],  # Maçã (peso baixo, acidez baixa)
    [150, 3.5],  # Caso limítrofe (pode ser difícil de classificar)
]

perceptron.test_new_cases(new_cases)

# Acurácia final
final_accuracy = perceptron.calculate_accuracy(X_train, y_train)
print(f"\nAcurácia final: {final_accuracy:.2%}")

