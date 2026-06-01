from __future__ import annotations

import argparse
import sys

from .constants import ORIGINATOR
from .oauth import CodexOAuth, CodexOAuthError


def _device_prompt(user_code: str, verification_uri: str) -> None:
    sys.stdout.write(
        f"\nTo sign in, open {verification_uri}\n"
        f"and enter the code: {user_code}\n\n"
    )
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codex-login",
        description="Sign in to OpenAI Codex (ChatGPT subscription) and store the "
        "credential in pi's auth.json.",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="Use the device-code flow instead of the browser flow.",
    )
    parser.add_argument(
        "--originator",
        default=ORIGINATOR,
        help="Originator sent on the authorize request (default: pi).",
    )
    parser.add_argument(
        "--auth-path",
        default=None,
        help="Override the auth.json path (defaults to pi's resolved path).",
    )
    args = parser.parse_args(argv)

    oauth = CodexOAuth(auth_path=args.auth_path, originator=args.originator)
    try:
        if args.device:
            oauth.device_login(on_prompt=_device_prompt)
        else:
            oauth.browser_login()
    except CodexOAuthError as exc:
        sys.stderr.write(f"Login failed: {exc}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\nLogin cancelled.\n")
        return 130

    sys.stdout.write("Login successful. Credential saved.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
