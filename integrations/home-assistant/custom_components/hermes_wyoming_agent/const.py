"""Constants for the Hermes Wyoming Agent integration."""

DOMAIN = "hermes_wyoming_agent"
CONF_NODE_RED_URL = "node_red_url"
CONF_NODE_RED_CREDENTIALS = "node_red_credentials"
CONF_STALL_ACK_SECONDS = "stall_ack_seconds"
CONF_STALL_ACK_TEXT = "stall_ack_text"
CONF_HERMES_URL = "hermes_url"
CONF_HERMES_CREDENTIALS = "hermes_credentials"

DEFAULT_NODE_RED_URL = "http://192.168.1.11:1880/endpoint/voice_in"
DEFAULT_NODE_RED_CREDENTIALS = "hermes_wyoming_nodered_auth"
DEFAULT_STALL_ACK_SECONDS = 0
DEFAULT_STALL_ACK_TEXT = "Accessing my databanks. This may take a moment."
DEFAULT_HERMES_URL = "http://192.168.1.10:8642/v1/chat/completions"
DEFAULT_HERMES_CREDENTIALS = "hermes_gateway_auth"
