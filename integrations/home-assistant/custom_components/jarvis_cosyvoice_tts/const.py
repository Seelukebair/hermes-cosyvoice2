"""Constants for the Jarvis CosyVoice TTS integration."""

DOMAIN = "jarvis_cosyvoice_tts"

CONF_BASE_URL = "base_url"
CONF_BEARER_TOKEN = "bearer_token"
CONF_FALLBACK_ENTITY_ID = "fallback_entity_id"
CONF_REQUEST_TIMEOUT = "request_timeout"

DEFAULT_BASE_URL = "http://192.168.1.10:17871"
DEFAULT_FALLBACK_ENTITY_ID = "tts.kokoro"
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_LANGUAGE = "en-US"
DEFAULT_VOICE = "default"

OPTION_SPEED = "speed"
OPTION_INSTRUCT = "instruct"
