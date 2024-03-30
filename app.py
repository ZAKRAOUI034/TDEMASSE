from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt 

app = Flask(__name__)

# Déclaration des variables globales
D_AB0 = None
D_BA0 = None

# Fonction pour saisir les données expérimentales
@app.route('/')
def index():
    return render_template('index.html')

# Fonction pour calculer D_AB
def calculate_D_AB(params, Xa, Xb, r_a, r_b, q_a, q_b, T):
    global D_AB0, D_BA0
    a_AB, a_BA = params

    # Calcul des termes de l'équation
    D = Xa * D_BA0 + Xb * np.log(D_AB0) + 2 * (
        Xa * np.log(Xa + (Xb * (r_b**(1/3))) / (r_a**(1/3))) +
        Xb * np.log(Xb + (Xa * (r_a**(1/3))) / (r_b**(1/3)))
    ) + 2 * Xa * Xb * (
        ((r_a**(1/3)) / (Xa * (r_a**(1/3)) + Xb * (r_b**(1/3)))) * (1 - ((r_a**(1/3)) / (r_b**(1/3)))) +
        ((r_b**(1/3)) / (Xa * (r_a**(1/3)) + Xb * (r_b**(1/3)))) * (1 - ((r_b**(1/3)) / (r_a**(1/3))))
    ) + Xb * q_a * (
        (1 - ((Xb * q_b * np.exp(-a_BA / T)) / (Xa * q_a + Xb * q_b * np.exp(-a_BA / T)))**2) * (-a_BA / T) +
        (1 - ((Xb * q_b) / (Xb * q_b + Xa * q_a * np.exp(-a_AB / T)))**2) * np.exp(-a_AB / T) * (-a_AB / T)
    ) + Xa * q_b * (
        (1 - ((Xa * q_a * np.exp(-a_AB / T)) / (Xa * q_a * np.exp(-a_AB / T) + Xb * q_b))**2) * (-a_AB / T) +
        (1 - ((Xa * q_a) / (Xa * q_a + Xb * q_b * np.exp(-a_BA / T)))**2) * np.exp(-a_BA / T) * (-a_BA / T)
    )
    # Calcul de D_AB
    D_AB = np.exp(D)

    return D_AB

# Fonction objectif pour la minimisation
def objective(params, D_AB_exp, T, Xa, Xb, r_a, r_b, q_a, q_b):
    D_AB_calculated = calculate_D_AB(params, Xa, Xb, r_a, r_b, q_a, q_b, T)
    return (D_AB_calculated - D_AB_exp)**2

# Fonction pour optimiser les paramètres
def optimize_parameters(D_AB_exp, T, Xa, Xb, r_a, r_b, q_a, q_b):
    # Paramètres initiaux
    params_initial = [1.0, 1.0]

    # Erreur initiale
    error = float('inf')

    # Tolerance
    tolerance = 1e-8

    # Nombre maximal d'itérations
    max_iterations = 1000
    iteration = 0

    start_time = time.time()  # Mesurer le temps d'exécution

    # Boucle d'ajustement des paramètres
    while error > tolerance and iteration < max_iterations:
        # Minimisation de l'erreur
        result = minimize(objective, params_initial, method='Nelder-Mead',
                          args=(D_AB_exp, T, Xa, Xb, r_a, r_b, q_a, q_b))

        # Paramètres optimisés
        a_AB_opt, a_BA_opt = result.x

        # Calcul de D_AB avec les paramètres optimisés
        D_AB_opt = calculate_D_AB([a_AB_opt, a_BA_opt], Xa, Xb, r_a, r_b, q_a, q_b, T)

        # Calcul de l'erreur
        error = abs(D_AB_opt - D_AB_exp)

        # Mise à jour des paramètres initiaux
        params_initial = [a_AB_opt, a_BA_opt]

        # Incrémentation du nombre d'itérations
        iteration += 1

    end_time = time.time()  # Fin du temps d'exécution
    execution_time = end_time - start_time  # Calcul du temps d'exécution

    # Affichage des résultats
    print("Paramètres optimisés:")
    print("a_AB =", a_AB_opt)
    print("a_BA =", a_BA_opt)
    print("D_AB calculé avec les paramètres optimisés:", D_AB_opt)
    print("Nombre d'itérations:", iteration)
    print("Temps d'exécution:", execution_time, "secondes")

    return a_AB_opt, a_BA_opt, D_AB_opt, iteration, execution_time


@app.route('/submit', methods=['POST'])
def submit():
    global D_AB0, D_BA0
    # Extraire les données du formulaire
    D_AB_exp = float(request.form['D_AB_exp'])
    T = float(request.form['T'])
    Xa = float(request.form['Xa'])
    Xb = 1 - Xa
    r_a = float(request.form['r_a'])
    r_b = float(request.form['r_b'])
    q_a = float(request.form['q_a'])
    q_b = float(request.form['q_b'])
    D_AB0 = float(request.form['D_AB0'])
    D_BA0 = float(request.form['D_BA0'])

    # Appeler optimize_parameters avec les entrées fournies
    a_AB_opt, a_BA_opt, D_AB_opt, iteration, execution_time = optimize_parameters(D_AB_exp, T, Xa, Xb, r_a, r_b, q_a, q_b)

    # Tracer D_AB en fonction de Xa jusqu'à Xa = 0.7
    Xa_values = np.linspace(0, 0.7, 10).tolist()  # Variation jusqu'à Xa = 0.7
    D_AB_values = [round(calculate_D_AB([a_AB_opt, a_BA_opt], xa, 1 - xa, r_a, r_b, q_a, q_b, T), 3) for xa in Xa_values]  # Arrondir à 3 chiffres après la virgule

    # Rendre le modèle résultat avec les paramètres optimisés
    return render_template('output.html', a_AB=a_AB_opt, a_BA=a_BA_opt, D_AB_opt=D_AB_opt, iteration=iteration, execution_time=execution_time, Xa_values=Xa_values, D_AB_values=D_AB_values)
