"""
🫁 PROJET : Détection de Pneumonie sur Radiographies Thoraciques
Utilisant TensorFlow et Deep Learning (CNN)

Auteur: Sara Merdas
Dataset: PneumoniaMNIST (MedMNIST v2)
Objectif: Classifier les radios en NORMAL vs PNEUMONIA
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = 'models/pneumonia_detector.h5'
RESULTS_DIR = 'results'

# Créer les répertoires s'ils n'existent pas
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "="*70)
print("🫁 DÉTECTION DE PNEUMONIE - ENTRAÎNEMENT DU MODÈLE")
print("="*70 + "\n")

# ============================================================================
# ÉTAPE 1 : CHARGER LES DONNÉES
# ============================================================================

print("📥 ÉTAPE 1: Chargement du dataset PneumoniaMNIST...")
print("-" * 70)

try:
    (ds_train, ds_val, ds_test), info = tfds.load(
        'pneumonia_mnist',
        split=['train', 'validation', 'test'],
        as_supervised=True,
        with_info=True
    )
    
    print("✅ Dataset chargé avec succès !")
    print(f"\n📊 Informations du dataset:")
    print(f"   • Images d'entraînement  : {info.splits['train'].num_examples:,}")
    print(f"   • Images de validation  : {info.splits['validation'].num_examples:,}")
    print(f"   • Images de test        : {info.splits['test'].num_examples:,}")
    print(f"   • Résolution            : 28×28 pixels")
    print(f"   • Type                  : Images en niveaux de gris")
    print(f"   • Classes               : NORMAL (0), PNEUMONIA (1)")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    print("Vérifiez votre connexion internet et réessayez.")
    exit(1)

# ============================================================================
# ÉTAPE 2 : PRÉTRAITEMENT DES DONNÉES
# ============================================================================

print("\n" + "-" * 70)
print("🔄 ÉTAPE 2: Prétraitement des données...")
print("-" * 70)

def preprocess(image, label):
    """
    Normalise les images de 0-255 à 0-1
    Ajoute une dimension de canal (28, 28) → (28, 28, 1)
    """
    image = tf.cast(image, tf.float32) / 255.0  # Normalisation
    image = tf.expand_dims(image, axis=-1)       # Ajouter dimension canal
    return image, label

# Appliquer le prétraitement
ds_train = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✅ Prétraitement terminé !")
print(f"   • Batch size : {BATCH_SIZE}")
print(f"   • Normalisation : images en [0, 1]")
print(f"   • Shuffle : activé pour l'entraînement")

# ============================================================================
# ÉTAPE 3 : CONSTRUCTION DU MODÈLE CNN
# ============================================================================

print("\n" + "-" * 70)
print("🤖 ÉTAPE 3: Construction du modèle CNN...")
print("-" * 70)

model = keras.Sequential([
    # ===== BLOC 1 =====
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
    layers.BatchNormalization(name='batch_norm_1'),
    layers.MaxPooling2D((2, 2), name='max_pool_1'),
    layers.Dropout(0.25, name='dropout_1'),
    
    # ===== BLOC 2 =====
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
    layers.BatchNormalization(name='batch_norm_2'),
    layers.MaxPooling2D((2, 2), name='max_pool_2'),
    layers.Dropout(0.25, name='dropout_2'),
    
    # ===== COUCHES DENSES =====
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='dense_1'),
    layers.BatchNormalization(name='batch_norm_3'),
    layers.Dropout(0.5, name='dropout_3'),
    
    # ===== COUCHE DE SORTIE =====
    layers.Dense(1, activation='sigmoid', name='output')
], name='PneumoniaDetector')

print("✅ Modèle créé !")
print("\n📐 Architecture du modèle:\n")
model.summary()

# ============================================================================
# ÉTAPE 4 : COMPILATION DU MODÈLE
# ============================================================================

print("\n" + "-" * 70)
print("⚙️  ÉTAPE 4: Compilation du modèle...")
print("-" * 70)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("✅ Modèle compilé !")
print(f"   • Optimiseur : Adam (lr=0.001)")
print(f"   • Loss : Binary Crossentropy")
print(f"   • Métriques : Accuracy, Precision, Recall, AUC")

# ============================================================================
# ÉTAPE 5 : ENTRAÎNEMENT
# ============================================================================

print("\n" + "-" * 70)
print("🚀 ÉTAPE 5: Entraînement du modèle...")
print("-" * 70)
print(f"   • Epochs : {EPOCHS}")
print(f"   • Batch size : {BATCH_SIZE}")
print("\n")

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    verbose=1
)

print("\n✅ Entraînement terminé !")

# ============================================================================
# ÉTAPE 6 : ÉVALUATION SUR L'ENSEMBLE DE TEST
# ============================================================================

print("\n" + "-" * 70)
print("📊 ÉTAPE 6: Évaluation sur l'ensemble de test...")
print("-" * 70)

test_results = model.evaluate(ds_test, verbose=0)
test_loss, test_acc, test_precision, test_recall, test_auc = test_results

print(f"\n✅ Résultats de test :")
print(f"   • Loss        : {test_loss:.4f}")
print(f"   • Accuracy    : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   • Precision   : {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"   • Recall      : {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"   • AUC-ROC     : {test_auc:.4f}")

# ============================================================================
# ÉTAPE 7 : PRÉDICTIONS DÉTAILLÉES & MÉTRIQUES
# ============================================================================

print("\n" + "-" * 70)
print("🔮 ÉTAPE 7: Prédictions détaillées...")
print("-" * 70)

# Récupérer tous les prédictions et labels du test set
y_true = []
y_pred = []
y_pred_proba = []

for images, labels in ds_test:
    predictions = model.predict(images, verbose=0)
    y_pred_proba.extend(predictions.flatten().tolist())
    y_pred.extend((predictions > 0.5).astype(int).flatten().tolist())
    y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

# Classification Report
print("\n📋 Rapport de classification détaillé:\n")
report = classification_report(
    y_true, y_pred,
    target_names=['NORMAL', 'PNEUMONIA'],
    digits=4
)
print(report)

# Sauvegarder le rapport
with open(f'{RESULTS_DIR}/metrics_report.txt', 'w') as f:
    f.write("RAPPORT DÉTAILLÉ - DÉTECTION DE PNEUMONIE\n")
    f.write("=" * 70 + "\n\n")
    f.write("RÉSULTATS DE TEST:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"Precision: {test_precision:.4f} ({test_precision*100:.2f}%)\n")
    f.write(f"Recall   : {test_recall:.4f} ({test_recall*100:.2f}%)\n")
    f.write(f"AUC-ROC  : {test_auc:.4f}\n")
    f.write(f"Loss     : {test_loss:.4f}\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-" * 70 + "\n")
    f.write(report)

print(f"✅ Rapport sauvegardé dans : {RESULTS_DIR}/metrics_report.txt")

# ============================================================================
# ÉTAPE 8 : VISUALISATIONS
# ============================================================================

print("\n" + "-" * 70)
print("📈 ÉTAPE 8: Génération des graphiques...")
print("-" * 70)

fig = plt.figure(figsize=(16, 12))

# 1. Accuracy
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['accuracy'], label='Train', linewidth=2, marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
ax1.set_title('Accuracy sur les Epochs', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Loss
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['loss'], label='Train', linewidth=2, marker='o')
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
ax2.set_title('Loss sur les Epochs', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Precision & Recall
ax3 = plt.subplot(2, 3, 3)
ax3.plot(history.history['precision'], label='Train Precision', linewidth=2, marker='o')
ax3.plot(history.history['val_precision'], label='Val Precision', linewidth=2, marker='s')
ax3.plot(history.history['recall'], label='Train Recall', linewidth=2, marker='^')
ax3.plot(history.history['val_recall'], label='Val Recall', linewidth=2, marker='v')
ax3.set_title('Precision & Recall', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Score')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Matrice de confusion
ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False,
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
ax4.set_title('Matrice de Confusion', fontsize=12, fontweight='bold')
ax4.set_ylabel('Réalité')
ax4.set_xlabel('Prédiction')

# 5. Courbe ROC
ax5 = plt.subplot(2, 3, 5)
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Hasard')
ax5.set_xlim([0.0, 1.0])
ax5.set_ylim([0.0, 1.05])
ax5.set_xlabel('Taux de Faux Positifs')
ax5.set_ylabel('Taux de Vrais Positifs')
ax5.set_title('Courbe ROC-AUC', fontsize=12, fontweight='bold')
ax5.legend(loc="lower right")
ax5.grid(alpha=0.3)

# 6. Résumé des métriques
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
metrics_text = f"""
RÉSUMÉ DES PERFORMANCES

Ensemble de Test ({len(y_true)} images)

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:    {test_acc*100:.2f}%
Precision:   {test_precision*100:.2f}%
Recall:      {test_recall*100:.2f}%
AUC-ROC:     {test_auc:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vrais Négatifs  : {cm[0, 0]}
Faux Positifs   : {cm[0, 1]}
Faux Négatifs   : {cm[1, 0]}
Vrais Positifs  : {cm[1, 1]}
"""
ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300, bbox_inches='tight')
print(f"✅ Graphiques sauvegardés dans : {RESULTS_DIR}/training_history.png")
plt.close()

# ============================================================================
# ÉTAPE 9 : SAUVEGARDE DU MODÈLE
# ============================================================================

print("\n" + "-" * 70)
print("💾 ÉTAPE 9: Sauvegarde du modèle...")
print("-" * 70)

model.save(MODEL_PATH)
print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

# ============================================================================
# ÉTAPE 10 : EXEMPLES DE PRÉDICTIONS
# ============================================================================

print("\n" + "-" * 70)
print("🔮 ÉTAPE 10: Exemples de prédictions...")
print("-" * 70 + "\n")

count = 0
for images, labels in ds_test.take(2):
    predictions = model.predict(images[:5], verbose=0)
    for i in range(min(5, len(images))):
        pred_proba = predictions[i][0]
        pred_label = "PNEUMONIA" if pred_proba > 0.5 else "NORMAL"
        true_label = "PNEUMONIA" if labels[i].numpy() == 1 else "NORMAL"
        confidence = max(pred_proba, 1 - pred_proba)
        
        match = "✅" if (pred_proba > 0.5) == labels[i].numpy() else "❌"
        print(f"{match} Image {count+1}:")
        print(f"   Prédiction : {pred_label} ({confidence*100:.2f}%)")
        print(f"   Réalité    : {true_label}")
        print()
        count += 1

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("=" * 70)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 70)
print(f"""
📊 RÉSUMÉ FINAL:

Dataset:
  • Images d'entraînement  : {info.splits['train'].num_examples:,}
  • Images de validation  : {info.splits['validation'].num_examples:,}
  • Images de test        : {info.splits['test'].num_examples:,}

Modèle:
  • Architecture          : CNN (2 blocs conv + 1 dense)
  • Paramètres            : {model.count_params():,}
  • Epochs                : {EPOCHS}
  • Batch size            : {BATCH_SIZE}

Résultats:
  • Accuracy              : {test_acc*100:.2f}%
  • Precision             : {test_precision*100:.2f}%
  • Recall                : {test_recall*100:.2f}%
  • AUC-ROC               : {test_auc:.4f}

Fichiers générés:
  ✅ {MODEL_PATH}
  ✅ {RESULTS_DIR}/training_history.png
  ✅ {RESULTS_DIR}/metrics_report.txt

Pour faire des prédictions, exécutez:
  python predict.py --image <chemin_image>

""")
print("=" * 70)
