#!/usr/bin/env python3
"""Assistant vocal Hermy : wake words, écoute continue, conversation et veille."""
from __future__ import annotations
import json
import argparse
import math
import os
import platform
import queue
import signal
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
HERMY_TTS_VOICE = os.getenv("HERMY_TTS_VOICE", "").strip()
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))
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

# Le mode graphique est facultatif et ne charge Qt qu'avec --gui.
GUI_MODE = False
GUI_PAUSED = threading.Event()
GUI_FORCE_LISTEN = threading.Event()
SHUTDOWN_REQUESTED = threading.Event()
GUI_EVENTS: queue.Queue[tuple[str, str]] = queue.Queue()
ACTIVE_TTS_PROCESS: subprocess.Popen | None = None


def gui_event(kind: str, text: str = "") -> None:
    if GUI_MODE:
        GUI_EVENTS.put((kind, text))


def request_gui_shutdown() -> None:
    global ACTIVE_TTS_PROCESS
    if SHUTDOWN_REQUESTED.is_set():
        return
    SHUTDOWN_REQUESTED.set()
    GUI_PAUSED.clear()
    if ACTIVE_TTS_PROCESS is not None and ACTIVE_TTS_PROCESS.poll() is None:
        ACTIVE_TTS_PROCESS.terminate()
    if MICROPHONE is not None:
        try:
            MICROPHONE.close()
        except Exception:
            pass


def wait_for_gui_resume() -> None:
    """Point de synchronisation commun entre les commandes GUI et l'assistant."""
    while GUI_MODE and GUI_PAUSED.is_set() and not SHUTDOWN_REQUESTED.is_set():
        GUI_PAUSED.wait(0.1)
    if GUI_MODE and SHUTDOWN_REQUESTED.is_set():
        raise KeyboardInterrupt


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
        command = ["say"]
        if HERMY_TTS_VOICE:
            command.extend(["-v", HERMY_TTS_VOICE])
        command.append(text)
        return command
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
    wait_for_gui_resume()
    command = speech_command(text)
    if command is None:
        return False
    try:
        gui_event("speaking", text)
        if GUI_MODE:
            global ACTIVE_TTS_PROCESS
            process = subprocess.Popen(command)
            ACTIVE_TTS_PROCESS = process
            try:
                while process.poll() is None:
                    if SHUTDOWN_REQUESTED.wait(0.1):
                        process.terminate()
                        process.wait()
                        raise KeyboardInterrupt
            finally:
                ACTIVE_TTS_PROCESS = None
        else:
            subprocess.run(command, check=False)
        return False
    except KeyboardInterrupt:
        raise
    except OSError as exc:
        print(f"(Synthèse vocale indisponible : {exc})")
        return False
    finally:
        gui_event("spoken", "")


class ContinuousMicrophone:
    """Capture le micro avec un seul flux ouvert pendant toute l'exécution."""

    def __init__(self, device: int | str | None = None) -> None:
        # File bornée : si Whisper/API prennent du temps, on conserve les blocs récents.
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self.pending = np.array([], dtype=np.float32)
        self.pending_lock = threading.Lock()
        self.last_callback_rms = 0.0
        self.callback_blocks = 0
        self.stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"(Audio : {status})", file=sys.stderr)
        block = indata[:, 0].copy()
        self.callback_blocks += 1
        if HERMY_DEBUG_AUDIO:
            self.last_callback_rms = float(np.sqrt(np.mean(np.square(block))))
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
    # Attend la reprise sans injecter de faux blocs silencieux dans la détection.
    wait_for_gui_resume()
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


def listen_for_wake(whisper: WhisperModel) -> bool:
    """Retourne True si le bouton GUI a demandé une écoute directe."""
    gui_event("status", "En veille — dis « Hermy »")
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

    def resume_directly(chunk: np.ndarray | None = None) -> bool:
        if not GUI_MODE or not GUI_FORCE_LISTEN.is_set():
            return False
        GUI_FORCE_LISTEN.clear()
        if chunk is not None and MICROPHONE is not None:
            MICROPHONE.prepend_audio(chunk)
        print("\n▶️ Veille désactivée : écoute directe depuis la mascotte.")
        gui_event("status", "Je t’écoute…")
        return True

    while True:
        if resume_directly():
            return True
        chunk = record_chunk(chunk_seconds)
        if resume_directly(chunk):
            return True
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        rms_tick += 1
        if HERMY_DEBUG_AUDIO and rms_tick >= 5:
            capture_rms = MICROPHONE.last_callback_rms if MICROPHONE is not None else 0.0
            print(
                f"\r🔎 RMS traité : {rms:.4f} | dernier bloc micro : {capture_rms:.4f}   ",
                end="", flush=True)
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
                    return False
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
            return False
        chunks = []
        pre_roll.clear()
        speaking = False
        silence = 0.0
        speech_seconds = 0.0


def listen_for_speech(whisper: WhisperModel, timeout: float, prompt: str) -> str:
    if GUI_MODE:
        GUI_FORCE_LISTEN.clear()
    gui_event("status", "Je t’écoute…")
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
        if GUI_MODE and GUI_FORCE_LISTEN.is_set():
            GUI_FORCE_LISTEN.clear()
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
    wait_for_gui_resume()
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


def run_assistant() -> None:
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
            device_index = sd.default.device[0] if GUI_MODE else None
            if isinstance(device_index, int) and device_index < 0:
                device_index = None
            device = sd.query_devices(device_index, "input") if device_index is not None else sd.query_devices(kind="input")
            print(
                f"🎚️ Micro d'entrée sélectionné : {device.get('name', 'inconnu')} ({device.get('max_input_channels', '?')} canaux)")
        except Exception as exc:
            device_index = None
            print(f"(Impossible d'identifier le micro : {exc})")
        MICROPHONE = ContinuousMicrophone(device_index)
        MICROPHONE.start()
        print("🎙️ Micro ouvert en continu (Ctrl+C pour le fermer).")
        if HERMY_DEBUG_AUDIO:
            print("Mode diagnostic audio actif : RMS = énergie sonore (pas une preuve de parole) ; VAD + validation stricte filtrent les faux réveils.")
        while True:
            direct_listen = listen_for_wake(whisper)
            spoken = listen_for_speech(
                whisper,
                COMMAND_TIMEOUT,
                "🎙️ Je t'écoute… parle après le bip." if not direct_listen
                else "🎙️ Écoute directe activée depuis la mascotte : je t'écoute…",
            )
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
            # La demande qui suit « nouvelle conversation » passe par un
            # chemin d'écoute dédié : traite aussi « Stop » ici, avant l'API.
            if normalize(spoken) in STOP_WORDS:
                print("⏹️ Arrêt demandé : retour en veille.")
                continue
            answer = local_ack_response(spoken)
            local_ack = answer is not None
            if answer is not None:
                print("Réponse locale (sans appel API)…")
            else:
                print("Hermy réfléchit…")
                gui_event("thinking")
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
                    gui_event("thinking")
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


def run_gui() -> None:
    """Fenêtre mascotte Qt, transparente et réservée à macOS."""
    if platform.system() != "Darwin":
        fail("--gui est disponible uniquement sur macOS.")
    try:
        from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
        from PySide6.QtGui import QColor, QFont, QPainter, QPen, QIcon, QPixmap, QLinearGradient, QBrush, QPainterPath
        from PySide6.QtWidgets import QApplication, QWidget, QMenu
    except ImportError:
        fail("PySide6 manque. Sur macOS, installe les dépendances avec : python -m pip install -r requirements.txt")

    class Mascot(QWidget):
        def __init__(self) -> None:
            super().__init__(None, Qt.WindowType.FramelessWindowHint |
                             Qt.WindowType.WindowStaysOnTopHint |
                             Qt.WindowType.Tool)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            # Qt.Tool est un NSPanel sur macOS et se masque à la désactivation
            # de l'application par défaut. Garde la mascotte visible hors focus.
            self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
            self.setMouseTracking(True)
            self.setFixedSize(190, 220)
            self.hover = False
            self.drag_offset = None
            self.speaking = False
            self.mode = "sleep"
            self.speech_text = ""
            self.speech_started = 0.0
            self.speech_duration = 1.0
            self.mouth_open = 0.0
            self.paused = False
            self.blink = False
            self.message = ""
            self.message_until = 0.0
            self.manual_status_until = 0.0
            self.load_position()
            self.show()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.animate)
            self.timer.start(60)

        def load_position(self) -> None:
            try:
                data = json.loads((ROOT / "hermy_gui_position.json").read_text())
                self.move(int(data["x"]), int(data["y"]))
            except (OSError, ValueError, KeyError, TypeError):
                screen = QApplication.primaryScreen().availableGeometry()
                self.move(screen.right() - self.width() - 40,
                          screen.bottom() - self.height() - 60)

        def save_position(self) -> None:
            try:
                (ROOT / "hermy_gui_position.json").write_text(
                    json.dumps({"x": self.x(), "y": self.y()}))
            except OSError:
                pass

        def animate(self) -> None:
            now = time.monotonic()
            self.blink = (now % 4.8) < 0.13
            if self.mode == "speaking":
                text = self.speech_text
                progress = min(max(0.0, now - self.speech_started) / self.speech_duration, 0.999)
                char_index = min(int(progress * max(1, len(text))), max(0, len(text) - 1))
                char = text[char_index].lower() if text else "a"
                shapes = {"a": 1.0, "à": 1.0, "â": 0.9, "o": 0.82, "ô": 0.82,
                          "e": 0.52, "è": 0.66, "ê": 0.66, "i": 0.30, "î": 0.30,
                          "u": 0.36, "û": 0.36, "y": 0.30, "œ": 0.7}
                target = 0.0 if char in ",.;:!?…" else shapes.get(char, 0.08)
                self.mouth_open += (target - self.mouth_open) * 0.38
            else:
                self.mouth_open *= 0.72
            if self.message and not self.speaking and now > self.message_until:
                self.message = ""
            self.update()

        def set_message(self, kind: str, text: str) -> None:
            if kind == "speaking":
                self.speaking = True
                self.mode = "speaking"
                self.speech_text = text
                self.speech_started = time.monotonic()
                self.speech_duration = max(1.0, len(text) * 0.058 + text.count(",") * 0.12 + text.count(".") * 0.2)
                self.message = ""
            elif kind == "spoken":
                self.speaking = False
                self.mode = "listening"
                self.message = ""
            elif kind == "status":
                if self.paused or time.monotonic() < self.manual_status_until:
                    return
                self.mode = "sleep" if "veille" in text.lower() else "listening"
                self.message = ""
            elif kind == "thinking":
                self.mode = "thinking"
                self.message = ""
            self.update()

        @staticmethod
        def inside_ellipse(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
            left, top, width, height = rect
            dx = (x - left - width / 2) / (width / 2)
            dy = (y - top - height / 2) / (height / 2)
            return dx * dx + dy * dy <= 1

        def on_character(self, point) -> bool:
            x, y = point.x(), point.y()
            return any(self.inside_ellipse(x, y, r) for r in (
                (45, 12, 43, 45), (101, 12, 43, 45), (32, 40, 126, 125),
                (22, 115, 55, 72), (113, 115, 55, 72), (48, 151, 87, 63)))

        def toggle_pause(self) -> None:
            if SHUTDOWN_REQUESTED.is_set():
                return
            self.paused = not self.paused
            if self.paused:
                GUI_PAUSED.set()
                self.mode = "sleep"
                self.message = ""
                print("\n⏸️ Veille activée depuis la mascotte.", flush=True)
            else:
                GUI_FORCE_LISTEN.set()
                if MICROPHONE is not None:
                    MICROPHONE.discard_buffer()
                GUI_PAUSED.clear()
                self.mode = "listening"
                self.message = ""
                print("\n▶️ Écoute reprise depuis la mascotte.", flush=True)
            if self.paused and MICROPHONE is not None:
                MICROPHONE.discard_buffer()
            self.message_until = time.monotonic() + 3
            self.manual_status_until = time.monotonic() + 2
            self.update()

        def quit_app(self) -> None:
            request_gui_shutdown()

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                p = event.position()
                if self.hover and self.inside_ellipse(p.x(), p.y(), (139, 70, 35, 35)):
                    self.toggle_pause()
                    return
                if self.hover and self.inside_ellipse(p.x(), p.y(), (139, 112, 35, 35)):
                    self.quit_app()
                    return
                if not self.on_character(p):
                    return
                self.drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
            elif event.button() == Qt.MouseButton.RightButton:
                menu = QMenu()
                menu.addAction("Reprendre" if self.paused else "Mettre en veille", self.toggle_pause)
                menu.addSeparator()
                menu.addAction("Quitter Hermy", self.quit_app)
                menu.exec(event.globalPosition().toPoint())

        def mouseMoveEvent(self, event) -> None:
            point = event.position()
            over_controls = (self.hover and (
                self.inside_ellipse(point.x(), point.y(), (139, 70, 35, 35)) or
                self.inside_ellipse(point.x(), point.y(), (139, 112, 35, 35))))
            self.hover = self.on_character(point) or over_controls
            if self.drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self.drag_offset)
            self.update()

        def mouseReleaseEvent(self, event) -> None:
            if self.drag_offset is not None:
                self.drag_offset = None
                self.save_position()

        def leaveEvent(self, event) -> None:
            self.hover = False
            self.update()

        def paintEvent(self, event) -> None:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            black, white = QColor("#24242b"), QColor("#fffaf1")
            now = time.monotonic()
            sleep_mode = self.mode == "sleep"
            listen_mode = self.mode == "listening"
            thinking_mode = self.mode == "thinking"
            bob = math.sin(now * (1.1 if sleep_mode else 1.55 if thinking_mode else 2.1)) * (1.2 if sleep_mode else 1.3 if thinking_mode else 1.8)
            ear_motion = math.sin(now * 5.5) * 1.3 if listen_mode else 0.0
            p.save()
            p.translate(0, bob)
            # Personnage vectoriel : toutes les zones non dessinées restent transparentes.
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(black)
            p.drawEllipse(QRectF(45, 12 - ear_motion, 43, 45))
            p.drawEllipse(QRectF(101, 12 + ear_motion, 43, 45))
            p.drawEllipse(QRectF(32, 40, 126, 125))
            p.drawEllipse(QRectF(22, 115, 55, 72)); p.drawEllipse(QRectF(113, 115, 55, 72))
            p.drawEllipse(QRectF(48, 151, 87, 63))
            p.setBrush(white)
            p.drawEllipse(QRectF(47, 35, 84, 83)); p.drawEllipse(QRectF(58, 150, 67, 52))
            # Bras/pattes séparés sous le visage.
            p.setBrush(black)
            p.drawEllipse(QRectF(28, 119, 40, 54)); p.drawEllipse(QRectF(122, 119, 40, 54))
            p.setBrush(QColor("#25242b"))
            p.drawEllipse(QRectF(57, 59, 31, 39)); p.drawEllipse(QRectF(96, 59, 31, 39))
            if sleep_mode or self.blink:
                p.setPen(QPen(white, 2.5))
                p.drawLine(66, 79, 79, 79); p.drawLine(105, 79, 118, 79)
            else:
                gaze = math.sin(now * 1.5) * 1.4 if listen_mode else 0.0
                gaze_y = -2.0 if thinking_mode else 0.0
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(white)
                p.drawEllipse(QRectF(67, 68, 13, 16)); p.drawEllipse(QRectF(106, 68, 13, 16))
                p.setBrush(black)
                p.drawEllipse(QRectF(71 + gaze, 72 + gaze_y, 5, 9)); p.drawEllipse(QRectF(110 + gaze, 72 + gaze_y, 5, 9))
                p.setBrush(white)
                p.drawEllipse(QRectF(72 + gaze, 73 + gaze_y, 2, 2)); p.drawEllipse(QRectF(111 + gaze, 73 + gaze_y, 2, 2))
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(black)
            p.drawEllipse(QRectF(86, 89, 12, 8))
            p.setPen(QPen(black, 2)); p.drawLine(92, 96, 92, 101)
            if sleep_mode:
                p.setPen(QPen(QColor("#5b4650"), 2)); p.drawArc(QRectF(88, 99, 9, 7), 200 * 16, 140 * 16)
                rise = (now % 2.4) * 11
                alpha = int(90 + 150 * abs(math.sin(now * math.pi / 2.4)))
                p.setPen(QColor(70, 83, 125, alpha)); p.setFont(QFont("", 13, QFont.Weight.Bold))
                p.drawText(137, int(43 - rise), "Zzz")
            else:
                mouth_height = 2.5 + self.mouth_open * 12 if self.mode == "speaking" else 3.0
                mouth_width = 10 + self.mouth_open * 5 if self.mode == "speaking" else 8.0
                p.setPen(QPen(QColor("#693f46"), 1))
                p.setBrush(QColor("#30252a") if self.mouth_open > 0.3 else QColor("#d28b84"))
                p.drawEllipse(QRectF(92 - mouth_width / 2, 103, mouth_width, mouth_height))
                if self.mode == "speaking" and self.mouth_open > 0.58:
                    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor("#df8f98"))
                    p.drawEllipse(QRectF(89, 110, 6, 3))
            if thinking_mode:
                # Patte posée près du menton et points qui montent en cadence.
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(black)
                p.drawEllipse(QRectF(111, 111, 20, 35))
                p.drawEllipse(QRectF(108, 103, 23, 22))
                p.setBrush(QColor("#fffaf1"))
                p.drawEllipse(QRectF(112, 106, 4, 3)); p.drawEllipse(QRectF(119, 104, 4, 3))
                phase = (now * 1.25) % 3.0
                for i, (x, y) in enumerate(((142, 55), (150, 44), (158, 33))):
                    strength = 0.35 + 0.65 * max(0.0, 1.0 - abs(phase - i) / 0.9)
                    alpha = int(70 + strength * 150)
                    p.setBrush(QColor(82, 91, 126, alpha))
                    p.drawEllipse(QRectF(x, y - 2 * strength, 5 + strength, 5 + strength))
            if listen_mode:
                pulse = int(55 + 85 * (0.5 + 0.5 * math.sin(now * 3.2)))
                p.setPen(QPen(QColor(92, 147, 170, pulse), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawArc(QRectF(19, 69, 28, 38), 70 * 16, 220 * 16)
                p.drawArc(QRectF(143, 69, 28, 38), 250 * 16, 220 * 16)
            if self.hover:
                p.setPen(QPen(QColor(30, 30, 35, 170), 1)); p.setBrush(QColor(255, 255, 255, 225))
                p.drawEllipse(QRectF(139, 70, 35, 35)); p.drawEllipse(QRectF(139, 112, 35, 35))
                p.setPen(QPen(black, 2))
                if self.paused:
                    p.setBrush(black); p.drawPolygon([QPoint(152, 80), QPoint(152, 95), QPoint(164, 87)])
                else:
                    p.drawLine(151, 80, 151, 95); p.drawLine(160, 80, 160, 95)
                p.drawLine(151, 124, 162, 135); p.drawLine(162, 124, 151, 135)
            p.restore()
            p.end()

    def create_app_icon():
        """Icône panda dessinée en code, utilisée dans le Dock macOS."""
        pixmap = QPixmap(128, 128)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 128, 128)
        gradient.setColorAt(0, QColor("#8f7cff"))
        gradient.setColorAt(1, QColor("#46c8bd"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(QRectF(4, 4, 120, 120), 29, 29)
        dark = QColor("#24242b")
        painter.setBrush(dark)
        painter.drawEllipse(QRectF(23, 18, 38, 40))
        painter.drawEllipse(QRectF(67, 18, 38, 40))
        painter.setBrush(QColor("#fffaf1"))
        painter.drawEllipse(QRectF(19, 31, 90, 83))
        painter.setBrush(dark)
        painter.drawEllipse(QRectF(31, 51, 31, 36))
        painter.drawEllipse(QRectF(66, 51, 31, 36))
        painter.setBrush(QColor("#fffaf1"))
        painter.drawEllipse(QRectF(42, 61, 10, 13))
        painter.drawEllipse(QRectF(77, 61, 10, 13))
        painter.setBrush(dark)
        painter.drawEllipse(QRectF(46, 65, 5, 7))
        painter.drawEllipse(QRectF(81, 65, 5, 7))
        painter.drawEllipse(QRectF(58, 82, 13, 9))
        painter.setPen(QPen(dark, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        smile = QPainterPath()
        smile.moveTo(64, 89)
        smile.cubicTo(58, 100, 49, 98, 47, 94)
        smile.moveTo(64, 89)
        smile.cubicTo(70, 100, 79, 98, 81, 94)
        painter.drawPath(smile)
        painter.end()
        return QIcon(pixmap)

    app = QApplication(sys.argv[:1])
    # macOS uses the Qt application name for the Dock tooltip when running
    # this script directly through Python.
    app.setApplicationName("Hermy")
    app.setApplicationDisplayName("Hermy")
    app_icon = create_app_icon()
    app.setWindowIcon(app_icon)
    app.setQuitOnLastWindowClosed(True)
    widget = Mascot()
    widget.setWindowIcon(app_icon)
    if widget.windowHandle() is not None:
        widget.windowHandle().setIcon(app_icon)
    worker = threading.Thread(target=run_assistant, name="hermy-assistant", daemon=True)

    def handle_sigint(signum, frame) -> None:
        request_gui_shutdown()

    previous_sigint = signal.signal(signal.SIGINT, handle_sigint)

    def poll_events() -> None:
        if SHUTDOWN_REQUESTED.is_set():
            app.quit()
            return
        while True:
            try:
                kind, text = GUI_EVENTS.get_nowait()
            except queue.Empty:
                break
            widget.set_message(kind, text)
    poller = QTimer()
    poller.timeout.connect(poll_events)
    poller.start(100)
    worker.start()
    app.exec()
    request_gui_shutdown()
    worker.join(timeout=0.5)
    signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant vocal Hermy")
    parser.add_argument("--gui", action="store_true", help="Affiche la mascotte graphique (macOS uniquement)")
    args = parser.parse_args()
    GUI_MODE = args.gui
    if GUI_MODE:
        run_gui()
    else:
        run_assistant()
