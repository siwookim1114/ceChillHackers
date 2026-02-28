import { useEffect, useMemo, useRef, useState } from "react";
import type { Intervention } from "../types";

type AvatarTutorProps = {
  intervention: Intervention | null;
};

export function AvatarTutor({ intervention }: AvatarTutorProps) {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [autoplay, setAutoplay] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsReady, setTtsReady] = useState(false);
  const lastSpokenKeyRef = useRef<string>("");

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setTtsReady(false);
      return;
    }
    setTtsReady(true);

    const loadVoices = () => {
      const found = window.speechSynthesis.getVoices();
      setVoices(found);
      if (found.length > 0) {
        const preferred =
          found.find((voice) => voice.lang.toLowerCase().startsWith("en")) ?? found[0];
        setSelectedVoice((prev) => prev || preferred.name);
      }
    };

    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
      window.speechSynthesis.cancel();
    };
  }, []);

  const speechText = useMemo(() => {
    if (!intervention) {
      return "";
    }
    return `${intervention.reason} ${intervention.tutor_message}`;
  }, [intervention]);

  const playSpeech = (text: string) => {
    if (!ttsReady || !text) {
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    const selected = voices.find((voice) => voice.name === selectedVoice);
    if (selected) {
      utterance.voice = selected;
    }
    utterance.rate = 1.03;
    utterance.pitch = 1.05;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  const stopSpeech = () => {
    if (!ttsReady) {
      return;
    }
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  useEffect(() => {
    if (!autoplay || !intervention || !speechText) {
      return;
    }
    const uniqueKey = `${intervention.created_at}-${intervention.level}`;
    if (lastSpokenKeyRef.current === uniqueKey) {
      return;
    }
    lastSpokenKeyRef.current = uniqueKey;
    playSpeech(speechText);
  }, [autoplay, intervention, speechText, ttsReady, selectedVoice, voices]);

  return (
    <section className="avatar-card">
      <div className="avatar-row">
        <div className={`avatar-face ${isSpeaking ? "speaking" : ""}`} aria-hidden="true">
          <span className="eye left" />
          <span className="eye right" />
          <span className="mouth" />
        </div>
        <div>
          <h4>Tutor Avatar</h4>
          <p className="muted">{ttsReady ? "Voice ready" : "Voice unavailable in this browser"}</p>
        </div>
      </div>

      <div className="action-row">
        <button
          className="btn-muted"
          onClick={() => playSpeech(speechText)}
          disabled={!ttsReady || !speechText}
          type="button"
        >
          Play Hint Voice
        </button>
        <button className="btn-muted" onClick={stopSpeech} disabled={!ttsReady} type="button">
          Stop
        </button>
      </div>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={autoplay}
          onChange={(event) => setAutoplay(event.target.checked)}
        />
        Auto-play new intervention
      </label>

      {voices.length > 0 && (
        <label>
          Voice
          <select value={selectedVoice} onChange={(event) => setSelectedVoice(event.target.value)}>
            {voices.map((voice) => (
              <option key={`${voice.name}-${voice.lang}`} value={voice.name}>
                {voice.name} ({voice.lang})
              </option>
            ))}
          </select>
        </label>
      )}
    </section>
  );
}
