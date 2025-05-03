# **🌟 Application Web de Reconnaissance du Langage des Signes**

Cette application web, développée avec Flask, permet la reconnaissance en temps réel du langage des signes à l'aide d'une webcam. Elle utilise **MediaPipe** pour le suivi des mains, un modèle d'apprentissage automatique pré-entraîné pour classer les gestes, et **pyttsx3** pour une synthèse vocale. L'application détecte des signes spécifiques (par exemple, "bonjour", "oui", "non", "paix") et les affiche sur un flux vidéo tout en les vocalisant.

## **🎯 Fonctionnalités**
- 🖐️ Reconnaissance en temps réel des gestes de la main via MediaPipe.
- 🤖 Prédiction de signes à l'aide d'un modèle d'apprentissage automatique.
- 📹 Diffusion vidéo en direct avec annotations des prédictions.
- 🗣️ Synthèse vocale pour les signes reconnus.
- 🌐 Interface web pour visualiser le flux vidéo et les prédictions.

## **📋 Prérequis**
- Python 3.8 ou supérieur
- Une webcam fonctionnelle
- Le fichier du modèle pré-entraîné (`model.p`) dans le répertoire racine ou bien le générer tout seul en exécutant le fichier **inference.py**

## **🛠️ Installation**

1. **Cloner le dépôt** (ou télécharger les fichiers) :
   ```bash
   git clone <url-du-dépôt>
   cd <nom-du-dossier>
   ```

2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sous Windows : venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install flask opencv-python mediapipe numpy pickle5 pyttsx3 flask-cors
   ```

4. **Vérifier le modèle pré-entraîné** :
   - Placez le fichier `model.p` (modèle scikit-learn sérialisé) dans le répertoire racine.
   - Le modèle doit être entraîné pour reconnaître les signes définis dans `labels_dict` : `bonjour`, `oui`, `non`, `paix`.

## **🚀 Utilisation**

1. **Lancer l'application Flask** :
   ```bash
   python app.py
   ```

2. **Accéder à l'interface web** :
   - Ouvrez un navigateur et allez à `http://localhost:5000`.
   - Le flux vidéo affiche l'entrée de la webcam avec les signes reconnus annotés.
   - Les prédictions apparaissent sous la vidéo et sont vocalisées.

3. **Effectuer des signes** :
   - Placez votre main devant la webcam pour effectuer un signe reconnu (`bonjour`, `oui`, `non`, `paix`).
   - L'application détecte le signe, l'affiche sur le flux vidéo et le vocalise.

4. **Arrêter l'application** :
   - Appuyez sur `Ctrl+C` dans le terminal pour arrêter le serveur Flask.

## **📂 Structure du Projet**
```
reconnaissance-langage-signes/
├── ai.py              
├── dataset.py              
├── inference.py             
├── train.py              
├── test.py             
├── templates             
    ├── index.html              
├── model.p            
├── README.md          
```

## **✋ Signes Reconnus**
L'application reconnaît les signes suivants (définis dans `labels_dict`) :
- `0` : Bonjour
- `1` : Oui
- `2` : Non
- `3` : Paix


## **🌈 Améliorations Futures**
- Ajouter la reconnaissance de nouveaux signes ou de séquences de signes.
- Afficher les scores de confiance des prédictions.
- Permettre à l'utilisateur de corriger les prédictions pour réentraîner le modèle.
- Optimiser les performances pour les appareils moins puissants.

