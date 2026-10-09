"""
Audio Alert Module using pygame.mixer for Drowsiness Detection System.

Features:
- Pure numpy sound synthesis (no external audio files needed).
- ALARM: Loud, harsh two-tone siren alternating ~900 Hz and ~1400 Hz (0.25s each),
  played in a non-blocking loop while driver eyes remain closed.
- WARNING: Short double beep at ~700 Hz (0.15s each, 0.1s gap), played once
  on fatigue trigger.
- Dedicated non-blocking channels to ensure zero drop in video loop FPS.
- Safe initialization: gracefully falls back to silent mode if no audio device exists.
- Volume control and mute/unmute toggle.
"""

import sys
import numpy as np


class AudioManager:
    """
    Manages non-blocking audio alerts for drowsy siren and fatigue warning.
    """

    def __init__(self, no_sound: bool = False, volume: float = 1.0):
        self.no_sound = no_sound
        self.volume = max(0.0, min(1.0, float(volume)))
        self.is_muted = False
        self.available = False
        self.alarm_sound = None
        self.warning_sound = None
        self.alarm_channel = None
        self.warning_channel = None
        self._alarm_playing = False

        if not self.no_sound:
            self._init_mixer()
        else:
            print("[*] Audio alerts disabled via --no-sound flag.", flush=True)

    def _init_mixer(self):
        """Safely initialize pygame.mixer and synthesize alert sounds."""
        try:
            import pygame
            # Initialize mixer with 44.1kHz, signed 16-bit, stereo, small buffer for low latency
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            if pygame.mixer.get_init() is None:
                raise RuntimeError("pygame.mixer.get_init() returned None")

            pygame.mixer.set_num_channels(8)
            self.alarm_channel = pygame.mixer.Channel(0)
            self.warning_channel = pygame.mixer.Channel(1)

            # Synthesize alert waveforms using numpy
            self.alarm_sound = self._create_alarm_sound()
            self.warning_sound = self._create_warning_sound()

            self.set_volume(self.volume)
            self.available = True
            print("[+] Audio alert system initialized successfully (pygame.mixer).", flush=True)
        except Exception as e:
            self.available = False
            print(f"[!] Warning: Audio device unavailable or failed to initialize: {e}", flush=True)
            print("[*] Continuing in silent mode without sound.", flush=True)

    @staticmethod
    def _create_alarm_sound():
        """
        Synthesize loud, harsh two-tone siren alternating ~900 Hz and ~1400 Hz (0.25s each).
        Uses a mixture of sine and odd square harmonics for an emergency siren tone.
        """
        sample_rate = 44100
        duration_per_tone = 0.25
        t = np.linspace(0, duration_per_tone, int(sample_rate * duration_per_tone), endpoint=False)

        # Tone 1: 900 Hz (exact integer cycles in 0.25s: 225 cycles)
        tone1 = 0.65 * np.sin(2.0 * np.pi * 900.0 * t) + 0.35 * np.sign(np.sin(2.0 * np.pi * 900.0 * t))

        # Tone 2: 1400 Hz (exact integer cycles in 0.25s: 350 cycles)
        tone2 = 0.65 * np.sin(2.0 * np.pi * 1400.0 * t) + 0.35 * np.sign(np.sin(2.0 * np.pi * 1400.0 * t))

        # Concatenate tones into a 0.5s seamless loop
        alarm_wave = np.concatenate([tone1, tone2])

        # Normalize and convert to 16-bit signed stereo
        alarm_int16 = (28000 * alarm_wave).astype(np.int16)
        stereo_alarm = np.column_stack([alarm_int16, alarm_int16])

        import pygame
        return pygame.sndarray.make_sound(stereo_alarm)

    @staticmethod
    def _create_warning_sound():
        """
        Synthesize short double beep: ~700 Hz, 0.15s each with 0.10s gap.
        Total duration = 0.40s.
        """
        sample_rate = 44100
        t_beep = np.linspace(0, 0.15, int(sample_rate * 0.15), endpoint=False)
        beep = np.sin(2.0 * np.pi * 700.0 * t_beep)

        # Apply a smooth 5ms linear fade in/out to prevent start/stop speaker pops
        fade_len = int(sample_rate * 0.005)
        env = np.ones_like(beep)
        env[:fade_len] = np.linspace(0.0, 1.0, fade_len)
        env[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
        beep = beep * env

        gap = np.zeros(int(sample_rate * 0.10), dtype=np.float32)
        warning_wave = np.concatenate([beep, gap, beep])

        warning_int16 = (28000 * warning_wave).astype(np.int16)
        stereo_warning = np.column_stack([warning_int16, warning_int16])

        import pygame
        return pygame.sndarray.make_sound(stereo_warning)

    def set_volume(self, volume: float):
        """Set volume level (clamped between 0.0 and 1.0)."""
        self.volume = max(0.0, min(1.0, float(volume)))
        if self.available:
            effective_vol = 0.0 if self.is_muted else self.volume
            if self.alarm_sound is not None:
                self.alarm_sound.set_volume(effective_vol)
            if self.warning_sound is not None:
                self.warning_sound.set_volume(effective_vol)

    def set_mute(self, muted: bool):
        """Set mute state."""
        self.is_muted = muted
        if self.is_muted:
            self.stop_alarm()
        self.set_volume(self.volume)

    def toggle_mute(self) -> bool:
        """Toggle mute state and return new state."""
        self.set_mute(not self.is_muted)
        return self.is_muted

    def start_alarm(self):
        """Start playing the alarm siren in a continuous non-blocking loop."""
        if not self.available or self.no_sound or self.is_muted:
            return
        if not self._alarm_playing:
            self._alarm_playing = True
            if self.alarm_channel is not None and self.alarm_sound is not None:
                self.alarm_channel.play(self.alarm_sound, loops=-1)

    def stop_alarm(self):
        """Immediately stop the alarm siren."""
        if not self.available:
            self._alarm_playing = False
            return
        if self._alarm_playing:
            self._alarm_playing = False
            if self.alarm_channel is not None:
                self.alarm_channel.stop()

    def play_warning(self):
        """Play the short double beep warning sound once (non-blocking)."""
        if not self.available or self.no_sound or self.is_muted:
            return
        if self.warning_channel is not None and self.warning_sound is not None:
            self.warning_channel.play(self.warning_sound, loops=0)

    def get_status_str(self) -> str:
        """Return UI display string for sound status."""
        if self.no_sound:
            return "SOUND: OFF (--no-sound)"
        if not self.available:
            return "SOUND: UNAVAILABLE"
        if self.is_muted:
            return "SOUND: MUTED"
        return f"SOUND: ON ({int(self.volume * 100)}%)"

    def close(self):
        """Safely stop channels and release mixer resources."""
        if self.available:
            try:
                import pygame
                self.stop_alarm()
                if self.warning_channel is not None:
                    self.warning_channel.stop()
                pygame.mixer.quit()
            except Exception:
                pass
            self.available = False
