import matplotlib.pyplot as plt

# Exemple de données
n_images = [5, 10, 20, 50, 100]
psnr = [20.1, 23.5, 26.8, 29.0, 30.2]
ssim = [0.72, 0.80, 0.85, 0.89, 0.91]

# Création du graphique
plt.figure(figsize=(8, 5))

# PSNR - courbe + points
plt.plot(n_images, psnr, color='blue', marker='o', label='PSNR')

# SSIM - courbe + points
plt.plot(n_images, ssim, color='green', marker='o', label='SSIM')

# Personnalisation
plt.title("Évolution des métriques NeRF en fonction du nombre d'images")
plt.xlabel("Nombre d'images")
plt.ylabel("Valeurs des métriques")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Affichage
plt.show()
