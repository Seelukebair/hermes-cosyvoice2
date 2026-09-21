# Jarvis CosyVoice TTS for Home Assistant

This repository contains two isolated, update-safe pieces:

- `custom_components/jarvis_cosyvoice_tts`: a Home Assistant config-entry TTS provider.
- `jarvis-cosyvoice-ha-proxy.py` plus its systemd unit: an authenticated, HA-only bridge from `192.168.1.10:17871` to Jarvis's loopback CosyVoice API at `127.0.0.1:17870`.

The Home Assistant provider accepts no caller-selected voice. At request time, the bridge reads the same `/srv/cosyvoice2/data/voice_profiles/state.json` state used by Hermes, ignores every unaccepted `session_candidate`, verifies that the accepted session/default profile is saved and transcript-verified under `profiles/`, and forwards that profile ID explicitly. This prevents the backend's `voice: default` preview selector from leaking a one-shot candidate into Home Assistant. No profile ID is copied into Home Assistant. If no accepted saved session/default profile exists, the bridge returns `503 voice_profile_unavailable` instead of silently using the backend's built-in reference voice. If CosyVoice is unavailable or rejects a request, the provider calls the existing `tts.kokoro` Home Assistant engine without initiating media playback.

## Safety properties

- No Home Assistant core files are modified.
- No CosyVoice, model, proxy-model, or GPU service is started by installation or validation.
- The bridge accepts only `192.168.1.11` and loopback clients and requires a dedicated bearer token.
- The bridge never logs authorization headers, payloads, or spoken text.
- The config flow validates `/proxy-health`; it does not synthesize audio.
- Backend-bound `/health` and `/synthesize` work is limited to two concurrent requests by default. Saturation fails immediately with `503 proxy_capacity_exceeded` and `Retry-After: 1`; `JARVIS_COSYVOICE_MAX_CONCURRENT_REQUESTS` can set another positive limit at startup.
- Kokoro remains a separate Home Assistant/Wyoming provider and is used only as request-time fallback.

## Installed paths

- HAOS: `/config/custom_components/jarvis_cosyvoice_tts/`
- Jarvis1: `/usr/local/lib/jarvis-cosyvoice-ha-proxy.py`
- Jarvis1: `/etc/systemd/system/jarvis-cosyvoice-ha-proxy.service`
- Jarvis1: `/etc/jarvis-cosyvoice-ha-proxy.env` (mode 0600)

## Bootstrap stock profile

`install_bundled_stock_profile.py` atomically installs the pinned CosyVoice demo
asset as a non-identifying saved profile and selects it as the default. It requires
an explicit backup directory and refuses an ID that collides with a preview
candidate. This gives Home Assistant a safe primary voice until the user reviews
and accepts a custom profile; it never promotes or edits a private candidate.

`configure_ha_pipeline.py` updates an Assist pipeline through Home Assistant's
supported WebSocket API. It writes a mode-0600 backup first, reads the result
back, and restores the original record automatically if verification fails.

## Rollback

1. Remove the `Jarvis CosyVoice` config entry in Home Assistant.
2. Remove `/config/custom_components/jarvis_cosyvoice_tts` (or restore its timestamped backup if one exists).
3. Restart Home Assistant Core and confirm `tts.jarvis_cosyvoice` is absent while `tts.kokoro` remains registered.
4. As the host administrator, disable and stop `jarvis-cosyvoice-ha-proxy.service`.
5. Restore timestamped copies of the proxy unit/script/environment if they existed; otherwise remove the three installed host paths and reload the service manager.
6. Optionally remove the dedicated deployment public key from the Advanced SSH & Web Terminal app options using the protected options backup recorded during deployment, then restart only that app.

Do not restore the Home Assistant partial backup unless file/config-entry rollback fails; a backup restore is broader than removing this integration.
