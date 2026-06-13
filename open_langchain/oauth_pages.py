from __future__ import annotations

SUCCESS_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>OpenAI authentication</title></head>
<body style="font-family: system-ui, sans-serif; padding: 2rem;">
<h1>Authentication complete</h1>
<p>OpenAI authentication completed. You can close this window.</p>
</body></html>"""


def error_html(message: str) -> str:
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>Authentication error</title></head>\n'
        '<body style="font-family: system-ui, sans-serif; padding: 2rem;">\n'
        "<h1>Authentication error</h1>\n"
        f"<p>{message}</p>\n"
        "</body></html>"
    )
