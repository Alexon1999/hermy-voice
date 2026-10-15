#!/usr/bin/env python3
"""Assistant vocal Hermy : wake words, écoute continue, conversation et veille."""
from __future__ import annotations
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any
import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
API_BASE = os.getenv(
    "HERMY_API_BASE", "https://hermes-api.ualexon.dev/p/hermy/v1").rstrip("/")
API_KEY = os.getenv("API_SERVER_KEY", "").strip()
MODEL = os.getenv("HERMY_MODEL", "hermy")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LANGUAGE = os.getenv("STT_LANGUAGE", "fr").strip() or None
STT_INITIAL_PROMPT = os.getenv("STT_INITIAL_PROMPT", "").strip()
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
WAKE_CHECK_SECONDS = float(os.getenv("WAKE_CHECK_SECONDS", "0.8"))
WAKE_WINDOW_SECONDS = float(os.getenv("WAKE_WINDOW_SECONDS", "2.4"))
COMMAND_TIMEOUT = float(os.getenv("COMMAND_TIMEOUT", "30"))
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "0.7"))
FOLLOWUP_TIMEOUT = float(os.getenv("FOLLOWUP_TIMEOUT", "30"))
WAKE_BEAM_SIZE = max(1, int(os.getenv("WAKE_BEAM_SIZE", "1")))
COMMAND_BEAM_SIZE = max(1, int(os.getenv("COMMAND_BEAM_SIZE", "1")))
WHISPER_CPU_THREADS = max(1, int(
    os.getenv("WHISPER_CPU_THREADS", str(max(1, (os.cpu_count() or 2) - 1)))))
ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.012"))
CLAP_MIN_RMS = float(os.getenv("CLAP_MIN_RMS", "0.04"))
CLAP_MIN_PEAK = float(os.getenv("CLAP_MIN_PEAK", "0.20"))
CLAP_MIN_CREST = float(os.getenv("CLAP_MIN_CREST", "5.0"))
CLAP_MIN_INTERVAL = float(os.getenv("CLAP_MIN_INTERVAL", "0.22"))
CLAP_PAIR_WINDOW = float(os.getenv("CLAP_PAIR_WINDOW", "0.85"))
HERMY_DEBUG_AUDIO = os.getenv("HERMY_DEBUG_AUDIO", "1").strip(
).lower() in ("1", "true", "yes", "on")
STATE_FILE = ROOT / "conversation.json"
WAKE_WORDS = ("salut hermy", "bonjour hermy", "hey hermy", "hermy")
STOP_WORDS = {"stop", "stop hermy", "hermy stop"}
HTTP = requests.Session()


def fail(message: str, code: int = 1) -> None:
    print(f"\nErreur : {message}", file=sys.stderr)
    raise SystemExit(code)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text).split())


LOCAL_ACKNOWLEDGEMENTS = {
    "merci", "merci beaucoup", "merci bien", "d accord merci",
    "d accord merci beaucoup", "ok merci", "ok merci beaucoup",
    "okay merci", "super merci", "parfait merci", "tres bien merci",
    "c est parfait merci", "c est bon merci", "c est tout merci",
}


def local_ack_response(text: str) -> str | None:
    """Répond localement aux remerciements seuls, sans appeler l'API."""
    return "Avec plaisir !" if normalize(text) in LOCAL_ACKNOWLEDGEMENTS else None


def save_conversation(conversation_id: str) -> None:
    STATE_FILE.write_text(json.dumps(
        {"conversation_id": conversation_id}, indent=2) + "\n", encoding="utf-8")
    try:
        STATE_FILE.chmod(0o600)
    except OSError:
        pass


def new_conversation_id() -> str:
    return f"hermy-voice-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"


def load_conversation() -> str:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        value = data.get("conversation_id")
        if isinstance(value, str) and value.strip():
            return value
    except (OSError, json.JSONDecodeError):
        pass
    value = new_conversation_id()
    save_conversation(value)
    return value


def beep(kind: str = "ready") -> None:
    """Petit signal sonore généré localement, sans fichier audio requis."""
    try:
        duration, frequency = (0.13, 880) if kind == "ready" else (0.22, 660)
        t = np.arange(int(SAMPLE_RATE * duration),
                      dtype=np.float32) / SAMPLE_RATE
        tone = (0.18 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        sd.play(tone, SAMPLE_RATE, blocking=True)
    except Exception as exc:
        print(f"(Signal sonore indisponible : {exc})")


def speech_command(text: str) -> list[str] | None:
    system = platform.system().lower()
    if system == "darwin" and shutil.which("say"):
        return ["say", text]
    if shutil.which("espeak-ng"):
        return ["espeak-ng", "-v", "fr", text]
    if shutil.which("espeak"):
        return ["espeak", "-v", "fr", text]
    if shutil.which("spd-say"):
        return ["spd-say", "-l", "fr", text]
    print("(Installe espeak-ng pour activer la voix sur Linux.)")
    return None


def speak(
    text: str,
    whisper: WhisperModel | None = None,
    stop_whisper: WhisperModel | None = None,
) -> bool:
    """Lit la réponse sans surveiller le micro ni interrompre la synthèse."""
    if not text.strip():
        return False
    command = speech_command(text)
    if command is None:
        return False
    try:
        subprocess.run(command, check=False)
        return False
    except KeyboardInterrupt:
        raise
    except OSError as exc:
        print(f"(Synthèse vocale indisponible : {exc})")
        return False


class ContinuousMicrophone:
    """Capture le micro avec un seul flux ouvert pendant toute l'exécution."""

    def __init__(self) -> None:
        # File bornée : si Whisper/API prennent du temps, on conserve les blocs récents.
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self.pending = np.array([], dtype=np.float32)
        self.pending_lock = threading.Lock()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"(Audio : {status})", file=sys.stderr)
        block = indata[:, 0].copy()
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            # Supprime le bloc le plus ancien pour éviter une latence qui s'accumule.
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                pass
            try:
                self.blocks.put_nowait(block)
            except queue.Full:
                pass

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    def discard_buffer(self) -> None:
        """Jette les sons déjà capturés (bip, wake word ou voix de synthèse).

        Le flux du micro reste ouvert : on évite de réécouter la phrase qui vient
        de déclencher Hermy ou la réponse qu'Hermy vient de prononcer.
        """
        while True:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                break
        with self.pending_lock:
            self.pending = np.array([], dtype=np.float32)

    def prepend_audio(self, audio: np.ndarray) -> None:
        """Restore audio already consumed by the interruption monitor."""
        if audio.size == 0:
            return
        with self.pending_lock:
            self.pending = np.concatenate(
                (audio.astype(np.float32, copy=False), self.pending))

    def read(self, seconds: float) -> np.ndarray:
        target = max(1, int(SAMPLE_RATE * seconds))
        pieces: list[np.ndarray] = []
        count = 0
        # Réutilise la partie restante du dernier bloc au lieu de la jeter : cela
        # évite de perdre des échantillons à chaque lecture courte de 0,2/0,5 s.
        with self.pending_lock:
            if self.pending.size:
                take = min(target, self.pending.size)
                pieces.append(self.pending[:take])
                self.pending = self.pending[take:].copy()
                count += take
        while count < target:
            try:
                block = self.blocks.get(timeout=max(2.0, seconds + 1.0))
            except queue.Empty:
                if not pieces:
                    raise RuntimeError(
                        "aucun son reçu du micro ; vérifie le périphérique d'entrée macOS")
                break
            needed = target - count
            take = min(needed, block.size)
            pieces.append(block[:take])
            count += take
            if take < block.size:
                with self.pending_lock:
                    self.pending = np.concatenate((block[take:], self.pending))
        audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return audio.astype(np.float32, copy=False)


MICROPHONE: ContinuousMicrophone | None = None


def record_chunk(seconds: float) -> np.ndarray:
    if MICROPHONE is None:
        raise RuntimeError("le micro continu n'est pas initialisé")
    return MICROPHONE.read(seconds)


def transcribe(
    audio: np.ndarray,
    whisper: WhisperModel,
    *,
    wake_mode: bool = False,
    stop_mode: bool = False,
) -> str:
    """Transcrit uniquement les segments que le VAD identifie comme de la parole.

    En veille, un indice lexical léger aide Whisper avec le nom propre « Hermy ».
    Il ne déclenche jamais à lui seul le réveil : VAD, contrôles de qualité et
    correspondance exacte du texte restent obligatoires.
    """
    if audio.size == 0:
        return ""
    try:
        # En veille, donne à Whisper un indice lexical limité au nom propre
        # « Hermy ». La détection reste stricte ensuite : l'indice ne suffit
        # jamais à déclencher l'assistant sans transcription exacte validée.
        transcription_options = {}
        if stop_mode:
            # Modèle rapide dédié à l'interruption pendant la synthèse vocale.
            transcription_options["initial_prompt"] = "Stop Hermy. Hermy stop."
        elif wake_mode:
            # Indice lexical réservé à la veille : ne pas biaiser les commandes.
            transcription_options["initial_prompt"] = (
                "Hermy. Salut Hermy. Bonjour Hermy. Hey Hermy."
            )
        elif STT_INITIAL_PROMPT:
            # Optionnel : noms propres/termes récurrents fournis par l'utilisateur.
            transcription_options["initial_prompt"] = STT_INITIAL_PROMPT
        # La veille reste stricte pour limiter les faux réveils. Pour les commandes,
        # un seuil VAD plus souple et un padding plus large préservent les syllabes faibles.
        vad_parameters = (
            {
                "threshold": 0.30,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 150,
                "speech_pad_ms": 80,
            }
            if stop_mode
            else
            {
                "threshold": 0.60,
                "min_speech_duration_ms": 300,
                "min_silence_duration_ms": 350,
                "speech_pad_ms": 120,
            }
            if wake_mode
            else {
                "threshold": 0.45,
                "min_speech_duration_ms": 200,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            }
        )
        segments, _ = whisper.transcribe(
            audio,
            language=LANGUAGE,
            vad_filter=True,
            vad_parameters=vad_parameters,
            # Une seule hypothèse accélère nettement la veille. Les commandes
            # gardent un décodage plus robuste, configurable dans .env.
            beam_size=1 if stop_mode else (
                WAKE_BEAM_SIZE if wake_mode else COMMAND_BEAM_SIZE),
            condition_on_previous_text=False,
            no_speech_threshold=0.85 if stop_mode else 0.60,
            log_prob_threshold=-1.3 if stop_mode else -1.0,
            compression_ratio_threshold=3.0 if stop_mode else 2.4,
            **transcription_options,
        )
        accepted = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            # Écarte les segments dont Whisper lui-même estime qu'ils ne sont
            # probablement pas de la parole, ou dont la reconnaissance est faible.
            if not stop_mode and getattr(segment, "no_speech_prob", 0.0) > 0.60:
                continue
            if not stop_mode and getattr(segment, "avg_logprob", 0.0) < -1.0:
                continue
            if not stop_mode and getattr(segment, "compression_ratio", 0.0) > 2.4:
                continue
            accepted.append(text)
        result = " ".join(accepted).strip()
        # Whisper peut parfois répéter le même texte sur un signal ambigu.
        # En veille, une phrase répétée est presque toujours un faux positif.
        if wake_mode:
            words = normalize(result).split()
            if len(words) > 4:
                return ""
        return result
    except Exception as exc:
        # Ne pas laisser une erreur VAD tuer la boucle du micro.
        print(f"\n(Transcription : {exc})")
        return ""


def wake_detected(text: str) -> bool:
    """Accepte seulement un wake word exact, jamais une simple ressemblance."""
    value = normalize(text)
    # Pas d'alias phonétiques (« Hermes », « Ermy », etc.) ni de mot-clé noyé
    # dans une phrase plus longue : ils déclenchaient Hermy sur des bruits.
    return value in WAKE_WORDS


def clap_detected(audio: np.ndarray) -> bool:
    """Repère un claquement bref via son énergie et son facteur de crête."""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    crest_factor = peak / max(rms, 1e-8)
    return (
        rms >= CLAP_MIN_RMS
        and peak >= CLAP_MIN_PEAK
        and crest_factor >= CLAP_MIN_CREST
    )


def listen_for_wake(whisper: WhisperModel) -> None:
    print("\n💤 Veille — dis « Salut Hermy », « Bonjour Hermy », « Hey Hermy » ou « Hermy », ou claque deux fois des mains.")
    # Détecte chaque prise de parole une fois, puis la transcrit après une
    # courte pause. La fenêtre glissante précédente retraitait plusieurs fois
    # le même audio et répétait les faux positifs dans le diagnostic.
    chunk_seconds = 0.2
    pre_roll: deque[np.ndarray] = deque(maxlen=2)
    chunks: list[np.ndarray] = []
    speaking = False
    silence = 0.0
    speech_seconds = 0.0
    rms_tick = 0
    last_clap_at: float | None = None
    clap_armed = True
    threshold = max(0.010, ENERGY_THRESHOLD * 0.75)
    max_speech_seconds = max(4.0, WAKE_WINDOW_SECONDS)
    while True:
        chunk = record_chunk(chunk_seconds)
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        rms_tick += 1
        if HERMY_DEBUG_AUDIO and rms_tick >= 5:
            print(
                f"\r🔎 Micro actif — niveau audio RMS : {rms:.4f}   ", end="", flush=True)
            rms_tick = 0
        is_clap = clap_detected(chunk)
        now = time.monotonic()
        if is_clap and clap_armed:
            if last_clap_at is not None:
                interval = now - last_clap_at
                if CLAP_MIN_INTERVAL <= interval <= CLAP_PAIR_WINDOW:
                    print("\n👏 Double claquement détecté : Hermy se réveille.")
                    beep("ready")
                    if MICROPHONE is not None:
                        MICROPHONE.discard_buffer()
                    return
            if last_clap_at is None or now - last_clap_at > CLAP_PAIR_WINDOW:
                last_clap_at = now
            clap_armed = False
        elif not is_clap:
            # Un nouveau pic n'est un second claquement que si un bloc calme
            # a séparé les deux impacts ; la réverbération d'un coup unique
            # ne doit pas former une fausse paire.
            if rms < max(ENERGY_THRESHOLD, CLAP_MIN_RMS * 0.5):
                clap_armed = True
            if last_clap_at is not None and now - last_clap_at > CLAP_PAIR_WINDOW:
                last_clap_at = None
        if not speaking:
            pre_roll.append(chunk)
            if rms >= threshold:
                speaking = True
                chunks = list(pre_roll)
                speech_seconds = len(chunks) * chunk_seconds
                silence = 0.0
                pre_roll.clear()
            continue

        chunks.append(chunk)
        speech_seconds += chunk_seconds
        if rms >= threshold:
            silence = 0.0
        else:
            silence += chunk_seconds

        if silence < 0.4 and speech_seconds < max_speech_seconds:
            continue

        audio = np.concatenate(chunks)
        if HERMY_DEBUG_AUDIO:
            print("\r" + " " * 72 + "\r", end="", flush=True)
        try:
            heard = transcribe(audio, whisper, wake_mode=True)
        except Exception as exc:
            print(f"\n(Détection vocale : {exc})")
            heard = ""
        # Les transcriptions non pertinentes restent silencieuses. En
        # particulier, « nouvelle conversation » ne réveille pas Hermy.
        if heard and wake_detected(heard):
            print(f"\n🔔 Wake word détecté : {heard}")
            beep("ready")
            # Élimine le wake word et le bip déjà capturés avant d'écouter
            # une nouvelle commande. Le micro reste ouvert en continu.
            if MICROPHONE is not None:
                MICROPHONE.discard_buffer()
            return
        chunks = []
        pre_roll.clear()
        speaking = False
        silence = 0.0
        speech_seconds = 0.0


def listen_for_speech(whisper: WhisperModel, timeout: float, prompt: str) -> str:
    if prompt:
        print(prompt)
    chunks: list[np.ndarray] = []
    # Conserve 400 ms avant le début détecté : les consonnes/syllabes initiales
    # sont parfois moins fortes que la suite de la phrase et étaient coupées.
    pre_roll: deque[np.ndarray] = deque(maxlen=2)
    started = False
    silence = 0.0
    elapsed = 0.0
    chunk_seconds = 0.2
    while elapsed < timeout:
        chunk = record_chunk(chunk_seconds)
        elapsed += chunk_seconds
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        if not started:
            pre_roll.append(chunk)
            if rms >= ENERGY_THRESHOLD:
                started = True
                chunks.extend(pre_roll)
                pre_roll.clear()
                silence = 0.0
        elif rms >= ENERGY_THRESHOLD:
            silence = 0.0
            chunks.append(chunk)
        else:
            chunks.append(chunk)
            silence += chunk_seconds
            if silence >= SILENCE_SECONDS:
                break
    if not started:
        return ""
    audio = np.concatenate(chunks)
    started_at = time.perf_counter()
    text = transcribe(audio, whisper)
    if text:
        print("📝 Transcription locale obtenue.")
        print(f"⏱️ Whisper : {time.perf_counter() - started_at:.1f} s")
    return text


def listen_until_command(whisper: WhisperModel, prompt: str) -> str:
    """Écoute une seule fenêtre de 30 s, sans répéter le prompt."""
    print(prompt)
    deadline = time.monotonic() + COMMAND_TIMEOUT
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        spoken = listen_for_speech(whisper, remaining, "")
        if spoken:
            return spoken
    return ""


def extract_response(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    output = payload.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                    parts.append(part["text"])
        if parts:
            return "\n".join(parts).strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    raise RuntimeError(
        "Réponse API sans texte reconnu. Vérifie le format /responses de ta version Hermes.")


def ask_hermy(text: str, conversation_id: str) -> str:
    started_at = time.perf_counter()
    response = HTTP.post(
        f"{API_BASE}/responses",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": MODEL, "input": text, "conversation": conversation_id, "instructions": (
            f"Tu es {MODEL}, l'assistant vocal personnel de l'utilisateur. "
            "Réponds toujours en français, avec des phrases naturelles et concises, "
            "adaptées à être lues à voix haute. Pour une tâche longue, ne prétends "
            "pas qu'elle est lancée ou terminée si tu n'en as pas la confirmation. "
            "Pose les questions nécessaires avant d'agir."
        )},
        timeout=(10, REQUEST_TIMEOUT),
    )
    if response.status_code >= 400:
        if response.status_code == 401:
            raise RuntimeError(
                "401 : clé API invalide. Vérifie API_SERVER_KEY dans .env.")
        if response.status_code == 524:
            raise RuntimeError(
                "Le serveur Hermes a mis trop de temps à répondre (HTTP 524). Répète ta demande pour réessayer.")
        if response.status_code in (502, 503, 504):
            raise RuntimeError(
                f"Le service Hermes est temporairement indisponible (HTTP {response.status_code}). Réessaie dans un instant.")
        if response.status_code == 429:
            raise RuntimeError(
                "L’API Hermes reçoit trop de demandes (HTTP 429). Réessaie dans un instant.")
        raise RuntimeError(
            f"Erreur de l’API Hermes (HTTP {response.status_code}).")
    try:
        payload = response.json()
        # print("payload", payload)
    except ValueError as exc:
        raise RuntimeError(
            "L’API Hermes a renvoyé une réponse illisible.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "L’API Hermes a renvoyé un format de réponse inattendu.")
    answer = extract_response(payload)
    print(f"⏱️ API : {time.perf_counter() - started_at:.1f} s")
    return answer


def ask_with_recovery(
    text: str,
    conversation_id: str,
    whisper: WhisperModel,
    stop_whisper: WhisperModel | None,
) -> str | None:
    """Affiche une erreur utile et laisse l'utilisateur répéter sa demande."""
    while True:
        try:
            return ask_hermy(text, conversation_id)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            if isinstance(exc, requests.Timeout):
                message = "Le délai d’attente de l’API Hermes est dépassé. Répète ta demande pour réessayer."
            elif isinstance(exc, requests.ConnectionError):
                message = "Impossible de joindre l’API Hermes. Vérifie la connexion puis répète ta demande."
            elif isinstance(exc, requests.RequestException):
                message = "La communication avec l’API Hermes a échoué. Répète ta demande pour réessayer."
            else:
                message = str(exc)
            print(f"\n⚠️ {message}")
            speak(message, whisper, stop_whisper)
            text = listen_for_speech(
                whisper,
                FOLLOWUP_TIMEOUT,
                f"👂 Répète ta demande dans les {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille.",
            )
            if not text:
                return None
            if normalize(text) in STOP_WORDS:
                print("⏹️ Arrêt demandé : retour en veille.")
                return None
            print(f"\nToi : {text}")


def main() -> None:
    if not API_KEY or API_KEY == "REMPLACE_PAR_TA_CLE":
        fail("renseigne API_SERVER_KEY dans le fichier .env.")
    if not API_BASE.startswith("https://"):
        print("⚠️ Attention : l'URL de l'API n'utilise pas HTTPS.")
    conversation_id = load_conversation()
    print("🐼 Hermy Voice — écoute continue (macOS + Raspberry Pi)")
    print(f"API : {API_BASE}\nConversation : {conversation_id}")
    print("Ctrl+C pour quitter. Le micro est traité localement ; seul le texte transcrit est envoyé à Hermes.")
    try:
        print(f"Chargement de Whisper ({WHISPER_MODEL})…")
        whisper = WhisperModel(
            WHISPER_MODEL,
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=1,
        )
    except Exception as exc:
        fail(f"impossible de charger Whisper : {exc}")
    # L'interruption vocale a été retirée : un seul modèle Whisper suffit.
    stop_whisper = None
    global MICROPHONE
    try:
        try:
            device = sd.query_devices(kind="input")
            print(
                f"🎚️ Micro d'entrée sélectionné : {device.get('name', 'inconnu')} ({device.get('max_input_channels', '?')} canaux)")
        except Exception as exc:
            print(f"(Impossible d'identifier le micro : {exc})")
        MICROPHONE = ContinuousMicrophone()
        MICROPHONE.start()
        print("🎙️ Micro ouvert en continu (Ctrl+C pour le fermer).")
        if HERMY_DEBUG_AUDIO:
            print("Mode diagnostic audio actif : RMS = énergie sonore (pas une preuve de parole) ; VAD + validation stricte filtrent les faux réveils.")
        while True:
            listen_for_wake(whisper)
            spoken = listen_for_speech(
                whisper, COMMAND_TIMEOUT, "🎙️ Je t'écoute… parle après le bip.")
            if not spoken:
                print("Aucune commande détectée : retour en veille.")
                continue
            print(f"\nToi : {spoken}")
            normalized = normalize(spoken)
            if normalized in STOP_WORDS:
                print("⏹️ Arrêt demandé : retour en veille.")
                continue
            if "nouvelle conversation" in normalized:
                conversation_id = new_conversation_id()
                save_conversation(conversation_id)
                print(f"🆕 Nouvelle conversation : {conversation_id}")
                spoken = normalized.replace(
                    "nouvelle conversation", "", 1).strip(" ,.!?;:")
                if not spoken:
                    speak("C'est parti pour une nouvelle conversation.")
                    spoken = listen_until_command(
                        whisper,
                        f"🎙️ Nouvelle conversation. Je t'écoute pendant {COMMAND_TIMEOUT:g} secondes…",
                    )
                    if not spoken:
                        print("Aucune demande en 30 secondes : retour en veille.")
                        continue
                    print(f"\nToi : {spoken}")
            answer = local_ack_response(spoken)
            local_ack = answer is not None
            if answer is not None:
                print("Réponse locale (sans appel API)…")
            else:
                print("Hermy réfléchit…")
                beep("thinking")
                answer = ask_with_recovery(
                    spoken, conversation_id, whisper, stop_whisper)
            if answer is None:
                continue
            print(f"\nHermy :\n{answer}\n")
            was_interrupted = speak(answer, whisper, stop_whisper)
            # macOS « say » est audible pendant que le micro reste ouvert.
            # Vide les blocs accumulés pendant la réponse pour éviter que
            # Hermy ne prenne sa propre voix pour la prochaine commande.
            if MICROPHONE is not None and not was_interrupted:
                MICROPHONE.discard_buffer()
            if local_ack:
                print("💤 Retour en veille.")
                continue
            # Fenêtre de suivi : pas besoin de redire le wake word immédiatement après la réponse.
            followup_prompt = (
                f"👂 Hermy est arrêté. Continue à parler pendant {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille."
                if was_interrupted else
                f"👂 Tu peux continuer à parler pendant {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille."
            )
            followup = listen_for_speech(
                whisper, FOLLOWUP_TIMEOUT, followup_prompt)
            if not followup:
                print("Aucune suite : retour en veille.")
                continue
            spoken = followup
            while spoken:
                normalized = normalize(spoken)
                if normalized in STOP_WORDS:
                    if was_interrupted:
                        was_interrupted = False
                        spoken = listen_for_speech(
                            whisper, COMMAND_TIMEOUT,
                            f"🎙️ Hermy est arrêté. Je t’écoute pendant {COMMAND_TIMEOUT:g} secondes…",
                        )
                        continue
                    print("⏹️ Arrêt demandé : retour en veille.")
                    break
                if was_interrupted and normalized.startswith("stop "):
                    spoken = normalized[4:].strip()
                    was_interrupted = False
                    if not spoken:
                        spoken = listen_for_speech(
                            whisper, COMMAND_TIMEOUT,
                            f"🎙️ Hermy est arrêté. Je t’écoute pendant {COMMAND_TIMEOUT:g} secondes…",
                        )
                        continue
                    normalized = normalize(spoken)
                if "nouvelle conversation" in normalized:
                    conversation_id = new_conversation_id()
                    save_conversation(conversation_id)
                    print(f"🆕 Nouvelle conversation : {conversation_id}")
                    spoken = normalized.replace(
                        "nouvelle conversation", "", 1).strip()
                    if not spoken:
                        speak("C'est parti pour une nouvelle conversation.")
                        spoken = listen_until_command(
                            whisper,
                            f"🎙️ Nouvelle conversation. Je t'écoute pendant {COMMAND_TIMEOUT:g} secondes…",
                        )
                        if not spoken:
                            print(
                                "Aucune demande en 30 secondes : retour en veille.")
                            break
                        continue
                print(f"\nToi : {spoken}")
                answer = local_ack_response(spoken)
                local_ack = answer is not None
                if answer is not None:
                    print("Réponse locale (sans appel API)…")
                else:
                    print("Hermy réfléchit…")
                    beep("thinking")
                    answer = ask_with_recovery(
                        spoken, conversation_id, whisper, stop_whisper)
                if answer is None:
                    break
                print(f"\nHermy :\n{answer}\n")
                was_interrupted = speak(answer, whisper, stop_whisper)
                if MICROPHONE is not None and not was_interrupted:
                    MICROPHONE.discard_buffer()
                if local_ack:
                    break
                followup_prompt = (
                    f"👂 Hermy est arrêté. Continue à parler pendant {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille."
                    if was_interrupted else
                    f"👂 Tu peux continuer à parler pendant {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille."
                )
                spoken = listen_for_speech(
                    whisper, FOLLOWUP_TIMEOUT, followup_prompt)
            print("💤 Retour en veille.")
    except KeyboardInterrupt:
        print("\nÀ bientôt !")
    except (EOFError, OSError, RuntimeError) as exc:
        fail(
            f"entrée audio interrompue : {exc}. Vérifie le micro et les permissions.")
    finally:
        if MICROPHONE is not None:
            try:
                MICROPHONE.close()
            except Exception:
                pass
            MICROPHONE = None


if __name__ == "__main__":
    main()
