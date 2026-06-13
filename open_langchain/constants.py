from __future__ import annotations

import os

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"

REDIRECT_URI = "http://localhost:1455/auth/callback"

DEVICE_USER_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = "https://auth.openai.com/codex/device"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60

SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
PKCE_METHOD = "S256"

BROWSER_LOGIN_METHOD = "browser"
DEVICE_CODE_LOGIN_METHOD = "device_code"

CALLBACK_HOST = os.environ.get("PI_OAUTH_CALLBACK_HOST", "127.0.0.1")
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
ORIGINATOR = "pi"

PROVIDER_ID = "openai-codex"
PROVIDER_NAME = "ChatGPT Plus/Pro (Codex Subscription)"

REQUEST_TIMEOUT = 600.0
REFRESH_TIMEOUT = 60.0
DEVICE_POLL_TIMEOUT = 30.0
MINIMUM_INTERVAL_MS = 1000
