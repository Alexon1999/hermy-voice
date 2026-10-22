# Hermy Voice


<img src="./docs/hermy.png" alt="Hermy Voice Logo" width="200" style="display:block; margin:auto;" />

C'est un projet assistant vocal local pour macOS et Raspberry Pi. Il communique avec l'agent Hermes via l'API Hermes pour gérer les conversations, mais la détection des wake words et la transcription des commandes se font entièrement en local grâce à Faster-Whisper. l’audio n’est pas envoyé. Le texte reconnu et l’identifiant de conversation sont envoyés à l’API Hermes en HTTPS.

ex: notre Client vocal Python utilisera le profil Hermes Agent `hermy`. 

Nous avons deux plateformes cibles : macOS et Raspberry Pi. 

Sur macOS, il peut fonctionner en deux modes : console ou interface graphique.
Sur Raspberry Pi, il fonctionne uniquement en mode console.


  
## Fonctionnement

- En veille, Hermy se réveille avec **« Salut Hermy »**, **« Bonjour Hermy »**, **« Hey Hermy »** ou **« Hermy »**.
- Il accepte aussi un **double claquement de mains** : deux claquements nets séparés de 0,22 à 0,85 seconde. Un claquement isolé ne suffit pas. Les seuils sont réglables dans `.env` (voir ci-dessous) ; le clavier et les coups sur une table peuvent encore provoquer des faux réveils selon le micro et la pièce.
- Après le réveil, tu as jusqu’à 30 secondes pour commencer une commande. Une pause de 0,7 seconde termine normalement la phrase et déclenche sa transcription locale.
- Après la réponse, Hermy reste disponible 30 secondes pour une demande de suivi sans nouveau wake word, puis retourne en veille.
- Dire **« nouvelle conversation »** crée un nouvel identifiant et ouvre une seule fenêtre d’écoute de 30 secondes pour la première demande. Les bruits que Whisper ne transcrit pas ne redémarrent pas le délai. Sans demande reconnue à l’expiration, Hermy retourne en veille.
- Les remerciements seuls pris en charge (par exemple « merci », « merci beaucoup » ou « d’accord, merci ») reçoivent une réponse locale « Avec plaisir ! », sans appel API, puis Hermy retourne en veille.
- Dire **« Stop »** pendant une période d’écoute met fin au tour et retourne en veille. Hermy ne surveille pas le micro pendant sa synthèse et termine sa réponse avant de reprendre l’écoute.
- La console affiche séparément le temps de transcription Whisper et le temps de réponse de l’API. Cela aide à distinguer le délai local du délai du modèle distant ou d’une recherche web.

> Les seuils de claquement sont acoustiques et dépendent du micro, du volume et de la pièce. Si le clavier ou la table réveille encore Hermy, augmente `CLAP_MIN_RMS`, `CLAP_MIN_PEAK` et `CLAP_MIN_CREST`. Si deux claquements ne sont pas détectés, baisse-les légèrement. Les wake words restent détectés par transcription locale Whisper, pas par un moteur spécialisé de mot-clé.

## Installation — macOS Apple Silicon

Python 3.14 est recommandé. `PySide6` est installé uniquement sur macOS par le fichier de dépendances ; il n'est pas nécessaire sur Raspberry Pi.

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

Si l’installation de `faster-whisper` échoue avec une version de Python trop récente, utilise Python 3.11 ou 3.12 dans un environnement virtuel.

## Configuration

Copie `.env.example` vers `.env`, renseigne `API_SERVER_KEY`, puis ajuste les paramètres au besoin. Ne publie pas `.env` et ne l’ajoute pas à Git : il contient la clé privée de l’API.

### API Hermes

| Variable          | Description                                             | Exemple                                     |
| ----------------- | ------------------------------------------------------- | ------------------------------------------- |
| `HERMY_API_BASE`  | URL de base de l’API Hermes.                            | `https://hermes-api.ualexon.dev/p/hermy/v1` |
| `API_SERVER_KEY`  | Clé privée du profil Hermes. Obligatoire.               | `REMPLACE_PAR_TA_CLE`                       |
| `HERMY_MODEL`     | Nom du profil/modèle envoyé à l’API.                    | `hermy`                                     |
| `REQUEST_TIMEOUT` | Délai maximal d’attente de la réponse API, en secondes. | `180`                                       |

### Voix de Hermy sur macOS

Hermy utilise la synthèse vocale intégrée à macOS (`say`) et conserve par défaut la voix système, comme avant. Pour installer ou essayer d'autres voix, ouvre **Réglages Système → Accessibilité → Contenu énoncé → Voix du système**, puis télécharge une voix française. Apple explique la sélection et le téléchargement des voix [dans son guide macOS](https://support.apple.com/en-me/guide/mac-help/mchlp2290/mac).

Tu peux choisir une voix installée en renseignant `HERMY_TTS_VOICE` dans `.env`. Pour afficher les noms exacts disponibles sur ton Mac :

```bash
say -v '?'
```

Exemple : `HERMY_TTS_VOICE="Eddy (Français (France))"`. Laisse la valeur vide pour utiliser la voix système par défaut. Cette option ne change pas la voix utilisée sur Raspberry Pi.

### Whisper

| Variable               | Description                                                                                                                                                                                         | Exemple  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `WHISPER_MODEL`        | Modèle utilisé pour les wake words et les commandes (`tiny`, `base`, `small`, `medium`, `large-v3`). Les modèles plus grands sont généralement plus précis mais plus lents et gourmands en mémoire. | `medium` |
| `STT_LANGUAGE`         | Langue parlée ; laisse vide pour la détection automatique.                                                                                                                                          | `fr`     |
| `STT_INITIAL_PROMPT`   | Indice facultatif pour les noms propres ou termes récurrents.                                                                                                                                       | vide     |
| `WHISPER_DEVICE`       | Périphérique Faster-Whisper : `cpu` ou `cuda` si compatible.                                                                                                                                        | `cpu`    |
| `WHISPER_COMPUTE_TYPE` | Type de calcul ; `int8` convient au CPU, `float16` aux GPU compatibles.                                                                                                                             | `int8`   |
| `WHISPER_CPU_THREADS`  | Nombre de threads CPU utilisés par Whisper.                                                                                                                                                         | `6`      |
| `WAKE_BEAM_SIZE`       | Nombre d’hypothèses pour les wake words ; `1` privilégie la vitesse.                                                                                                                                | `1`      |
| `COMMAND_BEAM_SIZE`    | Nombre d’hypothèses pour les commandes ; une valeur plus élevée peut améliorer la reconnaissance au prix du délai.                                                                                  | `1`      |

### Audio, veille et claquements

| Variable              | Description                                                                                                                                          | Exemple |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `SAMPLE_RATE`         | Fréquence d’échantillonnage audio en hertz.                                                                                                          | `16000` |
| `ENERGY_THRESHOLD`    | Niveau RMS minimal pour commencer à capturer une commande. Augmente-le si le bruit démarre l’enregistrement ; baisse-le si la voix n’est pas captée. | `0.012` |
| `SILENCE_SECONDS`     | Silence nécessaire pour considérer une phrase terminée. Une valeur basse réduit l’attente, mais peut couper les longues pauses dans une phrase.      | `0.7`   |
| `COMMAND_TIMEOUT`     | Délai pour commencer une commande après le réveil.                                                                                                   | `30`    |
| `FOLLOWUP_TIMEOUT`    | Délai pour une demande suivante après une réponse.                                                                                                   | `30`    |
| `CLAP_MIN_RMS`        | RMS minimal d’un impact candidat au claquement. Augmente-le pour rejeter les sons moins forts.                                                       | `0.04`  |
| `CLAP_MIN_PEAK`       | Pic minimal d’un impact candidat.                                                                                                                    | `0.20`  |
| `CLAP_MIN_CREST`      | Facteur de crête minimal, utilisé pour privilégier les sons brefs.                                                                                   | `5.0`   |
| `CLAP_MIN_INTERVAL`   | Intervalle minimal entre deux claquements.                                                                                                           | `0.22`  |
| `CLAP_PAIR_WINDOW`    | Intervalle maximal accepté entre les deux claquements.                                                                                               | `0.85`  |
| `WAKE_WINDOW_SECONDS` | Durée maximale configurée d’un segment de parole en veille (minimum effectif : 4 secondes).                                                          | `2.4`   |
| `WAKE_CHECK_SECONDS`  | Ancien réglage conservé pour compatibilité ; la veille actuelle est segmentée et ne l’utilise pas.                                                   | `0.8`   |
| `HERMY_DEBUG_AUDIO`   | Affiche (`1`) ou masque (`0`) le niveau RMS du micro en veille. Le RMS mesure l’énergie sonore, pas la présence certaine d’une voix.                 | `1`     |

## Lancement

```bash
source .venv/bin/activate
python hermy_voice.py
```

Sur macOS, tu peux lancer la mascotte flottante avec :

```bash
python hermy_voice.py --gui
```

La fenêtre graphique utilise PySide6. Elle affiche une mascotte dessinée directement dans le programme sur un fond transparent, sans bulle de transcription des réponses. Le personnage s’anime en veille, pendant l’écoute et pendant la parole. Fais glisser le personnage pour le placer ; sa position est mémorisée. Survole-le pour afficher les commandes veille/réveil et quitter. Réveiller Hermy depuis la mascotte lance directement une fenêtre d’écoute de commande, sans wake word. Un clic droit ouvre aussi le menu. Le mode console reste le mode par défaut ; le mode graphique est réservé à macOS. Sur Raspberry Pi, continue à lancer `python hermy_voice.py` sans installer PySide6.

Au premier lancement, Faster-Whisper télécharge le modèle configuré. Sur macOS, autorise le Terminal à utiliser le micro dans **Réglages Système → Confidentialité et sécurité → Microphone**. Sur Raspberry Pi, vérifie les périphériques avec :

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

`Ctrl+C` arrête Hermy et ferme le micro. Le micro reste ouvert en continu pendant l’exécution ; sur macOS, son voyant reste allumé.


## Exécution en arrière-plan au démarrage de macOS

Pour lancer Hermy automatiquement à chaque ouverture de session, macOS utilise un **LaunchAgent**. Il exécute directement le Python de l’environnement virtuel du projet, sans créer ni installer une application bundle.

Le lanceur `run_hermy_background.py` démarre Hermy et écrit sa sortie dans `hermy-launchd.log`. Ajoute `--gui` à sa commande pour activer le mode graphique (macOS). La rotation conserve au maximum quatre fichiers : le journal courant plafonne à 5 Mio et trois anciens journaux sont gardés (`hermy-launchd.log.1`, `.2`, `.3`).

Créer le fichier `~/Library/LaunchAgents/dev.ualexon.hermy-voice.plist` avec le contenu suivant, en remplaçant `/Users/alexon` par le chemin de ton dossier utilisateur si nécessaire :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.ualexon.hermy-voice</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/alexon/dev/projects/hermy-voice/.venv/bin/python</string>
    <string>/Users/alexon/dev/projects/hermy-voice/run_hermy_background.py</string>
    <string>--gui</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/alexon/dev/projects/hermy-voice</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
</dict>
</plist>
```

`ProgramArguments` lance le wrapper Python avec l’environnement virtuel du projet. Celui-ci démarre `hermy_voice.py --gui`, récupère ses sorties et erreurs, et applique la rotation du journal. `WorkingDirectory` permet au programme de retrouver les fichiers du projet et son `.env`. `RunAtLoad` le lance au chargement de la session. `KeepAlive` reste désactivé afin que quitter Hermy ferme réellement la mascotte au lieu de la faire réapparaître automatiquement. Pour la rouvrir avant la prochaine session, lance **Hermy** depuis Applications ; au prochain démarrage de session, `RunAtLoad` le lancera à nouveau.

Tu peux aussi relancer Hermy depuis **Applications** sans saisir de commande. Le petit lanceur `Hermy.app` appelle `launchctl` pour démarrer l’agent déjà chargé ; Hermy continue de tourner avec le Python de `.venv`. Pour (re)créer le lanceur sur ce Mac, compile le script AppleScript fourni :

```bash
osacompile -o /Applications/Hermy.app launch_hermy.applescript
```

Après avoir créé le fichier, valider sa syntaxe et charger l’agent dans la session actuelle :

```bash
plutil -lint ~/Library/LaunchAgents/dev.ualexon.hermy-voice.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.ualexon.hermy-voice.plist
```

Commandes de gestion :

```bash
# Vérifier l’état et la commande chargée
launchctl print gui/$(id -u)/dev.ualexon.hermy-voice

# Redémarrer Hermy après une modification
launchctl kickstart -k gui/$(id -u)/dev.ualexon.hermy-voice

# Arrêter et désactiver le lancement automatique
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/dev.ualexon.hermy-voice.plist
```

Pour le relancer ensuite, exécuter de nouveau `launchctl bootstrap` avec le chemin du fichier plist. La permission d’accès au microphone doit être accordée à l’exécutable Python utilisé par l’environnement virtuel dans **Réglages Système → Confidentialité et sécurité → Microphone**. Consulte `hermy-launchd.log` pour diagnostiquer un problème de démarrage.


### Exécution en arrière-plan au démarrage de Raspberry Pi

Sur Raspberry Pi OS, utilise un service utilisateur **systemd**. Le lanceur Python démarre Hermy en mode console et applique la même rotation : 5 Mio maximum pour `hermy-launchd.log`, avec trois archives (`.1`, `.2`, `.3`). Vérifie d’abord que l’installation décrite plus haut est terminée et que le projet est dans `/home/pi/hermy-voice` ; adapte ce chemin si ton nom d’utilisateur ou l’emplacement du dépôt diffère.

Créer le service utilisateur :

```bash
mkdir -p ~/.config/systemd/user
nano ~/.config/systemd/user/hermy-voice.service
```

Y mettre le contenu suivant, en adaptant `/home/pi` au besoin :

```ini
[Unit]
Description=Hermy Voice assistant
After=default.target

[Service]
Type=simple
WorkingDirectory=/home/pi/hermy-voice
ExecStart=/home/pi/hermy-voice/.venv/bin/python /home/pi/hermy-voice/run_hermy_background.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

Activer et démarrer le service :

```bash
systemctl --user daemon-reload
systemctl --user enable --now hermy-voice.service
```

Commandes utiles :

```bash
# Vérifier l’état
systemctl --user status hermy-voice.service

# Redémarrer après une modification
systemctl --user restart hermy-voice.service

# Arrêter temporairement
systemctl --user stop hermy-voice.service

# Désactiver le lancement automatique
systemctl --user disable --now hermy-voice.service

# Consulter le journal avec rotation
 tail -n 100 ~/hermy-voice/hermy-launchd.log
ls -lh ~/hermy-voice/hermy-launchd.log*
```

Par défaut, un service utilisateur démarre à l’ouverture de session. Pour le lancer au démarrage du Raspberry Pi même sans ouverture de session, active le *lingering* pour ton utilisateur :

```bash
sudo loginctl enable-linger "$USER"
```

Le service tourne sous ton compte utilisateur afin d’accéder à ses fichiers `.env`, son environnement virtuel et son micro. Si l’accès audio est refusé, ajoute l’utilisateur aux groupes audio si nécessaire, puis reconnecte-toi : `sudo usermod -aG audio "$USER"`.


## Diagnostic

- Si le RMS reste proche de `0.0000` quand tu parles, vérifie l’entrée choisie dans **Réglages Système → Son → Entrée** et les permissions micro.
- Si les wake words ne sont pas reconnus, vérifie `STT_LANGUAGE=fr`, essaie un modèle Whisper plus précis et parle distinctement.
- Si les coups sur la table ou le clavier déclenchent Hermy, augmente les trois seuils `CLAP_MIN_*` ; garde deux claquements séparés dans la fenêtre configurée.
- Si Hermy ne détecte pas tes claquements, baisse légèrement ces seuils et rapproche-toi du micro.
- Si une phrase se coupe pendant une pause, augmente `SILENCE_SECONDS`.
- Compare `⏱️ Whisper` et `⏱️ API` dans la console pour savoir si la transcription locale ou le service distant est la principale source de latence.
