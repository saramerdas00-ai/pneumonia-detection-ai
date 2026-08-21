"""
🔮 SCRIPT DE PRÉDICTION
Utilise le modèle entraîné pour prédire si une image est NORMAL ou PNEUMONIA
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

# Configuration
MODEL_PATH = 'models/pneumonia_detector.h5'

def load_model():
    """Charge le modèle entraîné"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Modèle non trouvé à {MODEL_PATH}")
        print("   Veuillez d'abord exécuter : python main.py")
        exit(1)
    
    print(f"📂 Chargement du modèle depuis {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Modèle chargé !")
    return model

def preprocess_image(image_path):
    """
    Charge et prétraite une image
    Redimensionne à 28x28 et normalise
    """
    try:
        # Charger l'image
        img = Image.open(image_path)
        
        # Convertir en niveaux de gris si nécessaire
        if img.mode != 'L':
            img = img.convert('L')
        
        # Redimensionner à 28x28
        img = img.resize((28, 28))
        
        # Convertir en array numpy
        img_array = np.array(img)
        
        # Normaliser
        img_array = img_array.astype(np.float32) / 255.0
        
        # Ajouter dimension batch et canal
        img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)
        
        return img_array, np.array(img)
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image : {e}")
        return None, None

def predict_single_image(model, image_path):
    """Prédit si une image est NORMAL ou PNEUMONIA"""
    
    print("\n" + "="*70)
    print("🔮 PRÉDICTION - DÉTECTION DE PNEUMONIE")
    print("="*70 + "\n")
    
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        print(f"❌ Erreur : Fichier non trouvé : {image_path}")
        return
    
    print(f"📸 Image : {image_path}")
    
    # Prétraiter l'image
    img_processed, img_original = preprocess_image(image_path)
    
    if img_processed is None:
        return
    
    # Faire la prédiction
    print("🤖 Analyse en cours...")
    prediction = model.predict(img_processed, verbose=0)
    confidence = prediction[0][0]
    
    # Interpréter le résultat
    if confidence > 0.5:
        predicted_class = "PNEUMONIA"
        predicted_label = 1
        confidence_display = confidence
    else:
        predicted_class = "NORMAL"
        predicted_label = 0
        confidence_display = 1 - confidence
    
    # Afficher les résultats
    print("\n" + "-"*70)
    print("📊 RÉSULTATS:")
    print("-"*70)
    print(f"Classe prédite  : {predicted_class}")
    print(f"Confiance       : {confidence_display*100:.2f}%")
    print(f"Probabilité     : {confidence:.4f}")
    print("-"*70)
    
    # Créer une visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Afficher l'image
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title(f'Image originale\n{os.path.basename(image_path)}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Afficher la prédiction
    ax_pred = axes[1]
    ax_pred.axis('off')
    
    # Texte de prédiction
    result_color = '#ff6b6b' if predicted_class == "PNEUMONIA" else '#51cf66'
    result_emoji = '⚠️' if predicted_class == "PNEUMONIA" else '✅'
    
    prediction_text = f"""
{result_emoji} PRÉDICTION

Classe : {predicted_class}
Confiance : {confidence_display*100:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Probabilités:
  • PNEUMONIA : {confidence*100:.2f}%
  • NORMAL    : {(1-confidence)*100:.2f}%
"""
    
    ax_pred.text(0.5, 0.5, prediction_text, fontsize=14, family='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.3, pad=1.5))
    
    plt.suptitle('Analyse de Radiographie - Détection de Pneumonie', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Sauvegarder le résultat
    output_path = 'prediction_result.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Résultat sauvegardé : {output_path}")
    
    plt.show()
    
    return predicted_class, confidence_display

def predict_batch(model, directory):
    """Prédit sur tous les fichiers PNG d'un répertoire"""
    
    print("\n" + "="*70)
    print("🔮 PRÉDICTIONS PAR LOT - DÉTECTION DE PNEUMONIE")
    print("="*70 + "\n")
    
    if not os.path.isdir(directory):
        print(f"❌ Erreur : Répertoire non trouvé : {directory}")
        return
    
    # Trouver tous les fichiers PNG
    image_files = [f for f in os.listdir(directory) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans : {directory}")
        return
    
    print(f"📁 Répertoire : {directory}")
    print(f"🖼️  Images trouvées : {len(image_files)}\n")
    
    results = []
    
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(directory, filename)
        
        img_processed, _ = preprocess_image(image_path)
        if img_processed is None:
            continue
        
        prediction = model.predict(img_processed, verbose=0)
        confidence = prediction[0][0]
        predicted_class = "PNEUMONIA" if confidence > 0.5 else "NORMAL"
        confidence_display = confidence if confidence > 0.5 else 1 - confidence
        
        results.append({
            'filename': filename,
            'class': predicted_class,
            'confidence': confidence_display
        })
        
        print(f"{idx:2d}. {filename:40s} → {predicted_class:10s} ({confidence_display*100:6.2f}%)")
    
    # Résumé
    print("\n" + "-"*70)
    pneumonia_count = sum(1 for r in results if r['class'] == "PNEUMONIA")
    normal_count = len(results) - pneumonia_count
    
    print(f"RÉSUMÉ:")
    print(f"  • PNEUMONIA : {pneumonia_count}/{len(results)}")
    print(f"  • NORMAL    : {normal_count}/{len(results)}")
    print("-"*70)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='🔮 Prédiction de Pneumonie sur Radiographies'
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Chemin vers une image PNG/JPG')
    parser.add_argument('--batch', type=str, default=None,
                        help='Chemin vers un répertoire contenant plusieurs images')
    parser.add_argument('--demo', action='store_true',
                        help='Mode démo (crée une fausse image pour tester)')
    
    args = parser.parse_args()
    
    # Charger le modèle
    model = load_model()
    
    # Mode démo
    if args.demo:
        print("\n" + "="*70)
        print("🎨 MODE DÉMO - Création d'une image de test")
        print("="*70 + "\n")
        
        # Créer une image de test
        demo_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        demo_path = 'demo_xray.png'
        Image.fromarray(demo_image).save(demo_path)
        print(f"✅ Image de test créée : {demo_path}")
        
        predict_single_image(model, demo_path)
    
    # Mode image unique
    elif args.image:
        predict_single_image(model, args.image)
    
    # Mode batch
    elif args.batch:
        predict_batch(model, args.batch)
    
    # Par défaut : afficher l'aide
    else:
        print("🔮 PRÉDICTION - DÉTECTION DE PNEUMONIE\n")
        print("Utilisation:")
        print("  1. Image unique   : python predict.py --image chemin/vers/image.png")
        print("  2. Lot d'images   : python predict.py --batch chemin/vers/dossier/")
        print("  3. Mode démo      : python predict.py --demo\n")
        print("Exemples:")
        print("  python predict.py --image data/xray.png")
        print("  python predict.py --batch ./test_images/")
        print("  python predict.py --demo")
