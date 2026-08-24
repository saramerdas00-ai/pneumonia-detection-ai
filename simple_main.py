"""
🫁 DÉTECTION DE PNEUMONIE - VERSION SIMPLIFIÉE
Modèle CNN sans dépendances compliquées
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Vérifier et installer les dépendances si nécessaire
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print("✅ TensorFlow trouvé !")
except ImportError:
    print("⚠️  Installation de TensorFlow...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

print("\n" + "="*70)
print("🫁 DÉTECTION DE PNEUMONIE - ENTRAÎNEMENT CNN")
print("="*70 + "\n")

# ============================================================
# ÉTAPE 1 : CRÉER LES DONNÉES DE TEST
# ============================================================

print("📊 ÉTAPE 1 : Création des données de test...")
print("-" * 70)

np.random.seed(42)

# Créer des images factices 28x28 (simulant des radiographies)
X_train = np.random.rand(300, 28, 28, 1).astype(np.float32)
y_train = np.random.randint(0, 2, 300)

X_val = np.random.rand(100, 28, 28, 1).astype(np.float32)
y_val = np.random.randint(0, 2, 100)

X_test = np.random.rand(100, 28, 28, 1).astype(np.float32)
y_test = np.random.randint(0, 2, 100)

print(f"✅ Données créées !")
print(f"   • Images d'entraînement : {X_train.shape[0]}")
print(f"   • Images de validation : {X_val.shape[0]}")
print(f"   • Images de test : {X_test.shape[0]}")
print(f"   • Résolution : 28×28 pixels\n")

# ============================================================
# ÉTAPE 2 : CONSTRUIRE LE MODÈLE CNN
# ============================================================

print("-" * 70)
print("🤖 ÉTAPE 2 : Construction du modèle CNN...")
print("-" * 70)

model = keras.Sequential([
    # ===== BLOC 1 =====
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1'),
    layers.MaxPooling2D((2, 2), name='max_pool_1'),
    layers.Dropout(0.25, name='dropout_1'),
    
    # ===== BLOC 2 =====
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
    layers.MaxPooling2D((2, 2), name='max_pool_2'),
    layers.Dropout(0.25, name='dropout_2'),
    
    # ===== COUCHES DENSES =====
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='dense_1'),
    layers.Dropout(0.5, name='dropout_3'),
    
    # ===== COUCHE DE SORTIE =====
    layers.Dense(1, activation='sigmoid', name='output')
], name='PneumoniaDetector')

print("✅ Modèle créé !")
print(f"   • Architecture : CNN (2 blocs conv + 1 dense)")
print(f"   • Paramètres totaux : {model.count_params():,}\n")

# ============================================================
# ÉTAPE 3 : COMPILER LE MODÈLE
# ============================================================

print("-" * 70)
print("⚙️  ÉTAPE 3 : Compilation du modèle...")
print("-" * 70)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèle compilé !")
print(f"   • Optimiseur : Adam (lr=0.001)")
print(f"   • Loss : Binary Crossentropy\n")

# ============================================================
# ÉTAPE 4 : ENTRAÎNER LE MODÈLE
# ============================================================

print("-" * 70)
print("🚀 ÉTAPE 4 : Entraînement du modèle...")
print("-" * 70)
print()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

print("\n✅ Entraînement terminé !\n")

# ============================================================
# ÉTAPE 5 : ÉVALUATION
# ============================================================

print("-" * 70)
print("📊 ÉTAPE 5 : Évaluation sur l'ensemble de test...")
print("-" * 70)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✅ Résultats de test :")
print(f"   • Loss        : {test_loss:.4f}")
print(f"   • Accuracy    : {test_acc:.4f} ({test_acc*100:.2f}%)\n")

# ============================================================
# ÉTAPE 6 : PRÉDICTIONS
# ============================================================

print("-" * 70)
print("🔮 ÉTAPE 6 : Prédictions sur le test set...")
print("-" * 70)

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Prédictions effectuées !")
print(f"\n📋 Matrice de confusion :")
print(f"   • Vrais Négatifs  : {cm[0, 0]}")
print(f"   • Faux Positifs   : {cm[0, 1]}")
print(f"   • Faux Négatifs   : {cm[1, 0]}")
print(f"   • Vrais Positifs  : {cm[1, 1]}\n")

# ============================================================
# ÉTAPE 7 : GÉNÉRATION DES GRAPHIQUES
# ============================================================

print("-" * 70)
print("📈 ÉTAPE 7 : Génération des graphiques...")
print("-" * 70)

fig = plt.figure(figsize=(16, 12))

# 1. ACCURACY
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['accuracy'], label='Train', linewidth=2, marker='o', markersize=5)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=5)
ax1.set_title('Accuracy par Epoch', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. LOSS
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['loss'], label='Train', linewidth=2, marker='o', markersize=5)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=5)
ax2.set_title('Loss par Epoch', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. MATRICE DE CONFUSION
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False,
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
ax3.set_title('Matrice de Confusion', fontsize=12, fontweight='bold')
ax3.set_ylabel('Réalité')
ax3.set_xlabel('Prédiction')

# 4. CLASSIFICATION REPORT
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
report = classification_report(y_test, y_pred, 
                               target_names=['NORMAL', 'PNEUMONIA'], 
                               digits=4)
ax4.text(0.05, 0.95, report, fontsize=10, family='monospace',
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax4.set_title('Rapport de Classification', fontsize=12, fontweight='bold')

# 5. RÉSUMÉ DES MÉTRIQUES
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
metrics_text = f"""
✅ RÉSUMÉ DES PERFORMANCES

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:      {test_acc*100:.2f}%
Loss:          {test_loss:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Matrice de Confusion:

  Vrais Négatifs  : {cm[0, 0]}
  Faux Positifs   : {cm[0, 1]}
  Faux Négatifs   : {cm[1, 0]}
  Vrais Positifs  : {cm[1, 1]}
"""
ax5.text(0.05, 0.95, metrics_text, fontsize=10, family='monospace',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# 6. DISTRIBUTION DES PRÉDICTIONS
ax6 = plt.subplot(2, 3, 6)
ax6.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='NORMAL (réel)', color='green', edgecolor='black')
ax6.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='PNEUMONIA (réel)', color='red', edgecolor='black')
ax6.set_title('Distribution des Prédictions', fontsize=12, fontweight='bold')
ax6.set_xlabel('Probabilité PNEUMONIA')
ax6.set_ylabel('Nombre')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("✅ Graphiques sauvegardés dans 'training_results.png'\n")
plt.show()

# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("="*70)
print(f"""
📊 RÉSULTATS FINAUX:

Dataset:
  • Images d'entraînement : 300
  • Images de validation : 100
  • Images de test : 100

Modèle:
  • Architecture : CNN (2 blocs conv + 1 dense)
  • Paramètres : {model.count_params():,}
  • Epochs : 10
  • Batch size : 32

Résultats:
  • Accuracy : {test_acc*100:.2f}%
  • Loss : {test_loss:.4f}
  
Fichiers générés:
  ✅ training_results.png

Les graphiques montrent :
  • Accuracy (train vs validation)
  • Loss (train vs validation)
  • Matrice de confusion
  • Rapport de classification
  • Résumé des performances
  • Distribution des prédictions
""")
print("="*70)
