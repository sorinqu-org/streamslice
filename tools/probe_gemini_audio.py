from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    audio_path = Path(sys.argv[1])
    token = os.environ["CLIPROXY_API_KEY"]
    audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    body = {
        "model": "gemini-3.6-flash-high",
        "messages": [
            {"role": "system", "content": "Ты транскрибатор. Верни только JSON."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            'Транскрибируй аудио. Верни {"words":[{"start":0,'
                            '"end":1,"text":"слово"}]}'
                        ),
                    },
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8081/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        print(response.read().decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
