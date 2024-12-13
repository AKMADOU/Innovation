

### 1. **Créer l'environnement virtuel en Python :**
Pour commencer, créez un environnement virtuel pour votre projet Python :
```bash
python -m venv innov
```

### 2. **Activer l'environnement virtuel :**
Pour activer l'environnement virtuel sous macOS ou Linux, utilisez la commande suivante :
```bash
source innov/bin/activate
```
Sous Windows, utilisez cette commande à la place :
```bash
innov\Scripts\activate
```

### 3. **Installer les bibliothèques nécessaires :**
Une fois l'environnement activé, vous pouvez installer les bibliothèques dont vous avez besoin pour votre projet. Voici les bibliothèques installées dans cet exemple :
```bash
pip install pandas scikit-learn openpyxl
```

### 4. **Installer Ollama en local :**
Pour installer Ollama, rendez-vous sur le site officiel et téléchargez la version correspondante à votre système d'exploitation :
[Ollama - Téléchargement](https://ollama.com/download)

### 5. **Lancer Ollama après installation :**
Une fois l'installation terminée, ouvrez votre terminal et exécutez les commandes suivantes pour démarrer Ollama :

- Démarrer le modèle `llama3.2` :
  ```bash
  ollama run llama3.2
  ```

- Télécharger le modèle `nomic-embed-text` :
  ```bash
  ollama pull nomic-embed-text
  ```


  - Pour lancer le code en ligne de commande
    ```bash
    python train.py
    ```
