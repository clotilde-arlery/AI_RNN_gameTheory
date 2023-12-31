import tensorflow as tf
import numpy as np
from random import *
import copy

def IA_clotilde(H_s=[], H_o=[]):
    # Pour que TensorFlow ne raconte pas toute sa vie dans le terminal
    tf.get_logger().setLevel('ERROR')

    my_history = copy.copy(H_s)
    other_history = copy.copy(H_o)
    for i in range(len(H_s)):
        if H_s[i] == 'c':
            my_history[i] = 0
        else:
            my_history[i] = 1
    for i in range(len(H_o)):
        if H_o[i] == 'c':
            other_history[i] = 0
        else:
            other_history[i] = 1

    def create_rnn_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 2)), 
            tf.keras.layers.SimpleRNN(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Configuration du modèle pour l'entrainement (minimisation des pertes avec optimizer, calcul d'erreur avec loss - mse : mean squared error, évaluation des performances avec metrics)
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def generate_data(num_samples=1000, sequence_length=100):
        X = np.random.randint(2, size=(num_samples, sequence_length, 2))
        Y = np.zeros((num_samples, sequence_length, 1))

        for i in range(num_samples):
            for j in range(1, sequence_length):
                if X[i, j-1, 0] == 0 and X[i, j-1, 1] == 0:  # Coopération mutuelle
                    Y[i, j, 0] = 1

                elif X[i, j-1, 0] == 1 and X[i, j-1, 1] == 0:  # Défaut de l'autre joueur
                    Y[i, j, 0] = 0

                elif X[i, j-1, 0] == 0 and X[i, j-1, 1] == 1:  # Défaut de l'IA
                    Y[i, j, 0] = 1  # L'IA gagne 3 points !!!!!

                elif X[i, j-1, 0] == 1 and X[i, j-1, 1] == 1:  # Défaut mutuel
                    Y[i, j, 0] = 0

        return X, Y

    # Entraînement du modèle
    model = create_rnn_model()
    X_train, Y_train = generate_data()
    model.fit(X_train, Y_train, epochs=10, batch_size=256)

    def rnn_based_strategy(my_history, other_history, model):
        if not my_history:
            return "c"  # on coopère par défaut
        
        my_history_array = np.array([[my_history[i], other_history[i]] for i in range(len(my_history))])
        input_data = np.expand_dims(my_history_array, axis=0)  # Ajout d'une dimension supplémentaire

        prediction = model.predict(input_data)
        action = "c" if prediction[0, -1] > 0.5 else "d"

        nb_d = 0
        for i in range(len(H_o)):
            nb_d += 1 if H_o[i] == 'd' else 0
        if nb_d / len(H_o) >= 0.35:
            action = 'd'
        
        # Ajout du bruit 
        if len(H_o) % 15 == 0:
            action = 'c' if action == 'd' else 'd'

        return action
    

    action_to_take = rnn_based_strategy(my_history, other_history, model)
    return action_to_take
