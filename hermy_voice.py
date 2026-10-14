#!/usr/bin/env python3
"""Assistant vocal Hermy : wake words, écoute continue, conversation et veille."""
from __future__ import annotations
import json, os, platform, queue, shutil, subprocess, sys, threading, time, unicodedata, uuid
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
API_BASE = os.getenv("HERMY_API_BASE", "https://hermes-api.ualexon.dev/p/hermy/v1").rstrip("/")
API_KEY = os.getenv("API_SERVER_KEY", "").strip()
MODEL = os.getenv("HERMY_MODEL", "hermy")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LANGUAGE = os.getenv("STT_LANGUAGE", "fr").strip() or None
STT_INITIAL_PROMPT = os.getenv("STT_INITIAL_PROMPT", "").strip()
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
WAKE_CHECK_SECONDS = float(os.getenv("WAKE_CHECK_SECONDS", "1.2"))
WAKE_WINDOW_SECONDS = float(os.getenv("WAKE_WINDOW_SECONDS", "3.2"))
COMMAND_TIMEOUT = float(os.getenv("COMMAND_TIMEOUT", "12"))
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "1.3"))
FOLLOWUP_TIMEOUT = float(os.getenv("FOLLOWUP_TIMEOUT", "8"))
ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.012"))
HERMY_DEBUG_AUDIO = os.getenv("HERMY_DEBUG_AUDIO", "1").strip().lower() in ("1", "true", "yes", "on")
STATE_FILE = ROOT / "conversation.json"
WAKE_WORDS = ("salut hermy", "bonjour hermy", "hey hermy", "hermy")


def fail(message: str, code: int = 1) -> None:
    print(f"\nErreur : {message}", file=sys.stderr)
    raise SystemExit(code)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def save_conversation(conversation_id: str) -> None:
    STATE_FILE.write_text(json.dumps({"conversation_id": conversation_id}, indent=2) + "\n", encoding="utf-8")
    try: STATE_FILE.chmod(0o600)
    except OSError: pass


def new_conversation_id() -> str:
    return f"hermy-voice-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"


def load_conversation() -> str:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        value = data.get("conversation_id")
        if isinstance(value, str) and value.strip(): return value
    except (OSError, json.JSONDecodeError): pass
    value = new_conversation_id(); save_conversation(value); return value


def beep(kind: str = "ready") -> None:
    """Petit signal sonore généré localement, sans fichier audio requis."""
    try:
        duration, frequency = (0.13, 880) if kind == "ready" else (0.22, 660)
        t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
        tone = (0.18 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        sd.play(tone, SAMPLE_RATE, blocking=True)
    except Exception as exc:
        print(f"(Signal sonore indisponible : {exc})")


def speak(text: str) -> None:
    if not text.strip(): return
    system = platform.system().lower()
    try:
        if system == "darwin" and shutil.which("say"):
            subprocess.run(["say", text], check=False)
        elif shutil.which("espeak-ng"):
            subprocess.run(["espeak-ng", "-v", "fr", text], check=False)
        elif shutil.which("espeak"):
            subprocess.run(["espeak", "-v", "fr", text], check=False)
        elif shutil.which("spd-say"):
            subprocess.run(["spd-say", "-l", "fr", text], check=False)
        else: print("(Installe espeak-ng pour activer la voix sur Linux.)")
    except OSError as exc: print(f"(Synthèse vocale indisponible : {exc})")


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
            try: self.blocks.get_nowait()
            except queue.Empty: pass
            try: self.blocks.put_nowait(block)
            except queue.Full: pass

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
                    raise RuntimeError("aucun son reçu du micro ; vérifie le périphérique d'entrée macOS")
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


def transcribe(audio: np.ndarray, whisper: WhisperModel, *, wake_mode: bool = False) -> str:
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
        if wake_mode:
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
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.60,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            **transcription_options,
        )
        accepted = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            # Écarte les segments dont Whisper lui-même estime qu'ils ne sont
            # probablement pas de la parole, ou dont la reconnaissance est faible.
            if getattr(segment, "no_speech_prob", 0.0) > 0.60:
                continue
            if getattr(segment, "avg_logprob", 0.0) < -1.0:
                continue
            if getattr(segment, "compression_ratio", 0.0) > 2.4:
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


def listen_for_wake(whisper: WhisperModel) -> None:
    print("\n💤 Veille — dis « Salut Hermy », « Bonjour Hermy », « Hey Hermy » ou « Hermy ».")
    chunks: deque[np.ndarray] = deque()
    max_samples = int(SAMPLE_RATE * WAKE_WINDOW_SECONDS)
    last_check = 0.0
    while True:
        chunks.append(record_chunk(0.5))
        while sum(len(c) for c in chunks) > max_samples:
            chunks.popleft()
        now = time.monotonic()
        if now - last_check < WAKE_CHECK_SECONDS: continue
        last_check = now
        audio = np.concatenate(list(chunks)) if chunks else np.array([], dtype=np.float32)
        if audio.size < SAMPLE_RATE: continue
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        if HERMY_DEBUG_AUDIO:
            print(f"\r🔎 Micro actif — niveau audio RMS : {rms:.4f}   ", end="", flush=True)
        # Whisper peut inventer un wake word sur du silence. Ne lance pas la
        # transcription de veille si le niveau est trop faible.
        wake_energy_threshold = max(0.010, ENERGY_THRESHOLD * 0.75)
        if rms < wake_energy_threshold:
            continue
        try: heard = transcribe(audio, whisper, wake_mode=True)
        except Exception as exc:
            print(f"\n(Détection vocale : {exc})"); continue
        if HERMY_DEBUG_AUDIO and heard:
            print(f"\n📝 Whisper entend : « {heard} »")
        if heard and wake_detected(heard):
            print(f"\n🔔 Wake word détecté : {heard}")
            beep("ready")
            # Élimine le wake word et le bip déjà capturés avant d'écouter
            # une nouvelle commande. Le micro reste ouvert en continu.
            if MICROPHONE is not None:
                MICROPHONE.discard_buffer()
            return


def listen_for_speech(whisper: WhisperModel, timeout: float, prompt: str) -> str:
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
            if silence >= SILENCE_SECONDS: break
    if not started:
        return ""
    audio = np.concatenate(chunks)
    print("📝 Transcription locale…")
    return transcribe(audio, whisper)


def extract_response(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip(): return payload["output_text"].strip()
    output = payload.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict): continue
            content = item.get("content", [])
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str): parts.append(part["text"])
        if parts: return "\n".join(parts).strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str): return message["content"].strip()
    if payload.get("error"): raise RuntimeError(str(payload["error"]))
    raise RuntimeError("Réponse API sans texte reconnu. Vérifie le format /responses de ta version Hermes.")


def ask_hermy(text: str, conversation_id: str) -> str:
    response = requests.post(f"{API_BASE}/responses", headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json={"model": MODEL, "input": text, "conversation": conversation_id}, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        if response.status_code == 401: raise RuntimeError("401 : clé API invalide. Vérifie API_SERVER_KEY dans .env.")
        raise RuntimeError(f"API HTTP {response.status_code}: {response.text[:1200]}")
    return extract_response(response.json())


def main() -> None:
    if not API_KEY or API_KEY == "REMPLACE_PAR_TA_CLE": fail("renseigne API_SERVER_KEY dans le fichier .env.")
    if not API_BASE.startswith("https://"): print("⚠️ Attention : l'URL de l'API n'utilise pas HTTPS.")
    conversation_id = load_conversation()
    print("🐼 Hermy Voice — écoute continue (macOS + Raspberry Pi)")
    print(f"API : {API_BASE}\nConversation : {conversation_id}")
    print("Ctrl+C pour quitter. Le micro est traité localement ; seul le texte transcrit est envoyé à Hermes.")
    try:
        print(f"Chargement de Whisper ({WHISPER_MODEL})…")
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    except Exception as exc: fail(f"impossible de charger Whisper : {exc}")
    global MICROPHONE
    try:
        try:
            device = sd.query_devices(kind="input")
            print(f"🎚️ Micro d'entrée sélectionné : {device.get('name', 'inconnu')} ({device.get('max_input_channels', '?')} canaux)")
        except Exception as exc:
            print(f"(Impossible d'identifier le micro : {exc})")
        MICROPHONE = ContinuousMicrophone()
        MICROPHONE.start()
        print("🎙️ Micro ouvert en continu (Ctrl+C pour le fermer).")
        if HERMY_DEBUG_AUDIO:
            print("Mode diagnostic audio actif : RMS = énergie sonore (pas une preuve de parole) ; VAD + validation stricte filtrent les faux réveils.")
        while True:
            listen_for_wake(whisper)
            spoken = listen_for_speech(whisper, COMMAND_TIMEOUT, "🎙️ Je t'écoute… parle après le bip.")
            if not spoken:
                print("Aucune commande détectée : retour en veille.")
                continue
            print(f"\nToi : {spoken}")
            normalized = normalize(spoken)
            if "nouvelle conversation" in normalized:
                conversation_id = new_conversation_id(); save_conversation(conversation_id)
                print(f"🆕 Nouvelle conversation : {conversation_id}")
                spoken = normalized.replace("nouvelle conversation", "", 1).strip(" ,.!?;:")
                if not spoken:
                    speak("C'est parti pour une nouvelle conversation.")
                    continue
            print("Hermy réfléchit…")
            beep("thinking")
            try: answer = ask_hermy(spoken, conversation_id)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                print(f"Erreur API : {exc}"); speak("Je n'ai pas réussi à contacter Hermy."); continue
            print(f"\nHermy :\n{answer}\n")
            speak(answer)
            # macOS « say » est audible pendant que le micro reste ouvert.
            # Vide les blocs accumulés pendant la réponse pour éviter que
            # Hermy ne prenne sa propre voix pour la prochaine commande.
            if MICROPHONE is not None:
                MICROPHONE.discard_buffer()
            # Fenêtre de suivi : pas besoin de redire le wake word immédiatement après la réponse.
            followup = listen_for_speech(whisper, FOLLOWUP_TIMEOUT, f"👂 Tu peux continuer à parler pendant {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille.")
            if not followup:
                print("Aucune suite : retour en veille.")
                continue
            spoken = followup
            while spoken:
                print(f"\nToi : {spoken}\nHermy réfléchit…")
                beep("thinking")
                try: answer = ask_hermy(spoken, conversation_id)
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    print(f"Erreur API : {exc}"); speak("Je n'ai pas réussi à contacter Hermy."); break
                print(f"\nHermy :\n{answer}\n"); speak(answer)
                if MICROPHONE is not None:
                    MICROPHONE.discard_buffer()
                spoken = listen_for_speech(whisper, FOLLOWUP_TIMEOUT, f"👂 Tu peux continuer à parler pendant {FOLLOWUP_TIMEOUT:g} secondes, sinon je retourne en veille.")
            print("💤 Retour en veille.")
    except KeyboardInterrupt: print("\nÀ bientôt !")
    except (EOFError, OSError, RuntimeError) as exc: fail(f"entrée audio interrompue : {exc}. Vérifie le micro et les permissions.")
    finally:
        if MICROPHONE is not None:
            try: MICROPHONE.close()
            except Exception: pass
            MICROPHONE = None

if __name__ == "__main__": main()
