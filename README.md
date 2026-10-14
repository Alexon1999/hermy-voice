# Hermy Voice — macOS + Raspberry Pi 5

Client vocal Python pour le profil Hermes Agent `hermy`, avec détection locale des wake words, écoute continue et retour automatique en veille.

## Comportement

- En veille, Hermy écoute les mots : **« Salut Hermy »**, **« Bonjour Hermy »**, **« Hey Hermy »** et **« Hermy »**.
- Un bip confirme le réveil. Hermy écoute ensuite la commande, jusqu'à 12 secondes par défaut.
- Après une période de silence, l'audio est transcrit localement par Faster-Whisper et seul le texte est envoyé à l'API Hermes en HTTPS.
- Un petit bip est joué pendant le traitement.
- Après la réponse vocale, Hermy attend jusqu'à 8 secondes une demande suivante sans exiger de nouveau wake word. S'il n'entend rien, il retourne en veille.
- `conversation.json` conserve l'identifiant de conversation. Dire « nouvelle conversation » démarre un nouveau fil local.

> **Limite à connaître :** les wake words sont détectés par transcription locale Whisper, pas par un moteur spécialisé de détection de mot-clé. Sur un Raspberry Pi, cela consomme plus de CPU et la reconnaissance de « Hermy » peut varier selon le micro, le bruit et le modèle. Le modèle `base` est un bon point de départ ; ajustez `ENERGY_THRESHOLD` si le micro ne détecte pas la voix.

## Installation — macOS Apple Silicon

Python 3.11 ou 3.12 est recommandé.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
brew install portaudio
python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
nano .env
```

## Installation — Raspberry Pi OS 64 bits / Raspberry Pi 5

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential portaudio19-dev libportaudio2 espeak-ng ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
nano .env
```

Si l'installation de `faster-whisper` échoue avec une version de Python trop récente, utilise Python 3.11 ou 3.12 dans un environnement virtuel.

## Configuration — fichier `.env`

Copie `.env.example` vers `.env`, puis renseigne tes paramètres :

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Le fichier `.env` contient tes réglages locaux et ta clé API. Il ne doit pas être publié, envoyé dans un chat ni ajouté à un dépôt Git. L’archive ne contient aucune clé personnelle.

### Paramètres API

| Variable | À quoi ça sert | Valeur par défaut / exemple |
|---|---|---|
| `HERMY_API_BASE` | URL HTTPS de base de l’API Hermes Agent. | `https://hermes-api.ualexon.dev/p/hermy/v1` |
| `API_SERVER_KEY` | Clé privée `API_SERVER_KEY` du profil `hermy` sur le VPS. Obligatoire. | `REMPLACE_PAR_TA_CLE` à remplacer |
| `HERMY_MODEL` | Nom du profil/modèle envoyé à l’API. | `hermy` |
| `REQUEST_TIMEOUT` | Nombre maximal de secondes pendant lesquelles le client attend la réponse de l’API avant expiration. | `180` |

### Reconnaissance vocale locale

| Variable | À quoi ça sert | Valeur par défaut / exemple |
|---|---|---|
| `WHISPER_MODEL` | Modèle Faster-Whisper utilisé pour reconnaître les wake words et transcrire la commande. `tiny` est plus léger ; `small` est généralement plus précis que `base`, au prix d’une consommation CPU et d’un délai plus élevés. | `base` |
| `STT_LANGUAGE` | Langue de transcription : `fr` pour le français, `en` pour l’anglais, etc. Une valeur vide permet la détection automatique. | `fr` |
| `STT_INITIAL_PROMPT` | Indice facultatif pour les commandes (noms propres ou termes récurrents). Laisser vide si inutile ; ne pas y mettre une phrase entière. | vide |
| `SAMPLE_RATE` | Fréquence d’échantillonnage audio, en hertz. | `16000` |

### Écoute, silence et veille

| Variable | À quoi ça sert | Valeur par défaut / exemple |
|---|---|---|
| `WAKE_CHECK_SECONDS` | Intervalle, en secondes, entre les analyses audio pendant la veille. Un intervalle plus court peut rendre la détection plus réactive, au prix de davantage de traitement. | `1.2` |
| `WAKE_WINDOW_SECONDS` | Durée de la fenêtre audio examinée pour reconnaître un wake word. Une fenêtre trop courte peut couper la phrase ; une fenêtre plus longue peut augmenter le délai de détection. | `3.2` |
| `COMMAND_TIMEOUT` | Durée maximale d’écoute de la commande après le bip de réveil. | `12` |
| `SILENCE_SECONDS` | Durée de silence consécutif qui indique que tu as terminé de parler et déclenche la transcription. | `1.3` |
| `FOLLOWUP_TIMEOUT` | Temps laissé après une réponse pour prononcer une nouvelle demande sans redire le wake word. Sans parole détectée, Hermy retourne en veille. | `8` |
| `ENERGY_THRESHOLD` | Seuil d’énergie sonore qui évite de traiter les blocs presque silencieux. Ce seuil ne distingue pas à lui seul la voix du clavier : le VAD et la validation de transcription font ce filtrage ensuite. | `0.012` |

Les quatre wake words sont définis dans le code : « Salut Hermy », « Bonjour Hermy », « Hey Hermy » et « Hermy ». Ils ne sont pas configurables via `.env` dans cette version.

**Filtrage anti-faux réveil :** Faster-Whisper applique maintenant un VAD avant la transcription, sans lui suggérer les mots de réveil ; Hermy n’accepte que les quatre formulations exactes et rejette les transcriptions répétées. Cela réduit les faux positifs, sans pouvoir garantir zéro erreur dans tous les environnements. Pour améliorer la précision des commandes, définis `WHISPER_MODEL=small` dans `.env` (le premier lancement téléchargera ce modèle). La capture conserve maintenant 400 ms avant le début détecté et le VAD des commandes est moins agressif pour éviter de couper les syllabes faibles. Tu peux définir `STT_INITIAL_PROMPT` avec quelques noms propres/termes récurrents, mais un prompt trop long peut biaiser la reconnaissance. Si Hermy coupe ta phrase trop tôt, augmente `SILENCE_SECONDS` ; si le réveil ne reconnaît pas le mot-clé, essaie une valeur plus grande pour `WAKE_WINDOW_SECONDS` ou un modèle Whisper plus précis. Sur Raspberry Pi 5, les modèles plus grands peuvent ralentir la reconnaissance.

## Lancement

```bash
source .venv/bin/activate
python hermy_voice.py
```

Au premier lancement, Faster-Whisper télécharge le modèle configuré. macOS : autorise le Terminal à utiliser le micro dans Réglages Système → Confidentialité et sécurité → Microphone. Raspberry Pi : vérifie les périphériques audio avec `python -c "import sounddevice as sd; print(sd.query_devices())"`.

## Réglages utiles

- `FOLLOWUP_TIMEOUT=8` : temps laissé pour enchaîner après une réponse.
- `COMMAND_TIMEOUT=12` : délai maximal d'attente de la commande après le réveil.
- `SILENCE_SECONDS=1.3` : silence qui termine une phrase.
- `ENERGY_THRESHOLD=0.012` : seuil de détection de voix ; baisser légèrement si Hermy ne détecte pas une voix normale, augmenter si le bruit ambiant déclenche l'écoute.
- `WAKE_WINDOW_SECONDS=3.2` : fenêtre audio analysée pour reconnaître le wake word.

`Ctrl+C` arrête Hermy.

## Microphone continu

Le micro reste ouvert avec un flux audio unique pendant toute l’exécution, y compris en veille. Sur macOS, son voyant reste allumé tant que Hermy Voice tourne. `Ctrl+C` ferme le flux.


## Diagnostic si Hermy ne réagit pas au mot de réveil

Par défaut, `HERMY_DEBUG_AUDIO=1` affiche le nom du périphérique d'entrée, le niveau RMS capté et les textes reconnus par Whisper pendant la veille.

- Si le niveau RMS reste proche de `0.0000` même quand tu parles, macOS utilise probablement la mauvaise entrée ou n'envoie pas de signal au programme. Vérifie **Réglages Système → Son → Entrée** et **Réglages Système → Confidentialité et sécurité → Microphone**.
- Si le niveau RMS varie quand tu parles mais qu'aucune ligne `Whisper entend` n'apparaît, essaie `WHISPER_MODEL=small` ou vérifie la langue avec `STT_LANGUAGE=fr`.
- Si Whisper affiche tes mots mais pas `Wake word détecté`, copie la transcription affichée : le mot « Hermy » a peut-être été transcrit différemment.
- Une fois le problème résolu, tu peux désactiver ces messages avec `HERMY_DEBUG_AUDIO=0`.
