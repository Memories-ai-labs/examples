"""Thin HTTP client over the memories.ai Visual Search + Visual Intelligence APIs.

Only wraps the endpoints actually used by the example agents in this repo:

- POST /upload, POST /upload_url            (Visual Search — index your footage)
- GET  /get_metadata                        (Visual Search — poll indexing status)
- POST /search                              (Visual Search — semantic search, BY_CLIP / BY_AUDIO)
- GET  /search_audio_transcripts            (Visual Search — exact-phrase transcript search)
- POST /download                            (Visual Search — original-file URL)
- POST /vu/chat/completions                 (Visual Intelligence — Gemini / Qwen / Nova VLM)

Every method returns the parsed JSON `data` field, raising MemoriesError on a
non-2xx HTTP status or a non-"0000" business code. See ../../api-docs for the
full endpoint reference.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

import requests

DEFAULT_API_HOST = "https://api.memories.ai/serve/api/v1"
DEFAULT_VLM_HOST = "https://mavi-backend.memories.ai/serve/api/v2"
DEFAULT_VLM_MODEL = "gemini:gemini-2.5-flash"

# code=0001 is a generic "transient network abnormal" error returned by the
# Visual Search edge — usually one quick retry succeeds.
TRANSIENT_BUSINESS_CODES = {"0001"}


class MemoriesError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None, payload: Any = None):
        super().__init__(message)
        self.code = code
        self.payload = payload


class MemoriesClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_host: str | None = None,
        vlm_host: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 60.0,
        media_url_map: dict[str, str] | None = None,
    ) -> None:
        key = api_key or os.environ.get("MEMORIES_API_KEY")
        if not key:
            raise MemoriesError("MEMORIES_API_KEY is not set")
        self.api_key = key
        self.api_host = (api_host or os.environ.get("MEMORIES_API_HOST") or DEFAULT_API_HOST).rstrip("/")
        self.vlm_host = (vlm_host or os.environ.get("MEMORIES_VLM_HOST") or DEFAULT_VLM_HOST).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.media_url_map = dict(media_url_map or {})
        self.max_retries = 3

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    def _unwrap(self, resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            raise MemoriesError(
                f"HTTP {resp.status_code} from {resp.url}: {resp.text[:400]}",
                payload=resp.text,
            )
        try:
            body = resp.json()
        except ValueError as e:
            raise MemoriesError(f"Non-JSON response from {resp.url}: {resp.text[:400]}") from e
        code = body.get("code")
        if code is not None and code != "0000":
            raise MemoriesError(
                f"API error from {resp.url}: code={code} msg={body.get('msg')!r}",
                code=code,
                payload=body,
            )
        return body.get("data", body)

    def _post_with_retry(self, url: str, **kwargs: Any) -> Any:
        return self._call_with_retry("POST", url, **kwargs)

    def _get_with_retry(self, url: str, **kwargs: Any) -> Any:
        return self._call_with_retry("GET", url, **kwargs)

    def _call_with_retry(self, method: str, url: str, **kwargs: Any) -> Any:
        last_err: MemoriesError | None = None
        for attempt in range(self.max_retries):
            try:
                if method == "POST":
                    resp = self.session.post(url, **kwargs)
                else:
                    resp = self.session.get(url, **kwargs)
                return self._unwrap(resp)
            except MemoriesError as e:
                if e.code in TRANSIENT_BUSINESS_CODES and attempt < self.max_retries - 1:
                    last_err = e
                    time.sleep(0.4 * (2 ** attempt))
                    continue
                raise
        assert last_err is not None
        raise last_err

    # -------- Visual Search: upload + index ----------------------------------

    def upload_file(
        self,
        file_path: str,
        *,
        unique_id: str = "default",
        callback: str | None = None,
        datetime_taken: str | None = None,
        camera_model: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        tags: Iterable[str] | None = None,
        retain_original_video: bool = True,
        video_transcription_prompt: str | None = None,
        mime_type: str = "video/mp4",
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "unique_id": unique_id,
            "retain_original_video": str(retain_original_video).lower(),
        }
        if callback:
            data["callback"] = callback
        if datetime_taken:
            data["datetime_taken"] = datetime_taken
        if camera_model:
            data["camera_model"] = camera_model
        if latitude is not None:
            data["latitude"] = latitude
        if longitude is not None:
            data["longitude"] = longitude
        if tags:
            data["tags"] = list(tags)
        if video_transcription_prompt:
            data["video_transcription_prompt"] = video_transcription_prompt

        with open(file_path, "rb") as fh:
            resp = self.session.post(
                f"{self.api_host}/upload",
                headers=self._headers,
                files={"file": (os.path.basename(file_path), fh, mime_type)},
                data=data,
                timeout=self.timeout,
            )
        return self._unwrap(resp)

    def upload_url(
        self,
        url: str,
        *,
        unique_id: str = "default",
        callback: str | None = None,
        datetime_taken: str | None = None,
        camera_model: str | None = None,
        tags: Iterable[str] | None = None,
        video_transcription_prompt: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"url": url, "unique_id": unique_id}
        if callback:
            data["callback"] = callback
        if datetime_taken:
            data["datetime_taken"] = datetime_taken
        if camera_model:
            data["camera_model"] = camera_model
        if tags:
            data["tags"] = list(tags)
        if video_transcription_prompt:
            data["video_transcription_prompt"] = video_transcription_prompt
        resp = self.session.post(
            f"{self.api_host}/upload_url",
            headers=self._headers,
            data=data,
            timeout=self.timeout,
        )
        return self._unwrap(resp)

    def get_metadata(self, video_no: str) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.api_host}/get_metadata",
            headers=self._headers,
            params={"video_no": video_no},
            timeout=self.timeout,
        )
        return self._unwrap(resp)

    # -------- Visual Search: search ------------------------------------------

    def search(
        self,
        query: str,
        *,
        search_type: str = "BY_CLIP",
        unique_id: str = "default",
        top_k: int = 10,
        filtering_level: str | None = "medium",
        video_nos: Iterable[str] | None = None,
        tag: str | None = None,
        camera_tag: str | None = None,
        datetime_taken: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "search_param": query,
            "search_type": search_type,
            "unique_id": unique_id,
            "top_k": top_k,
        }
        if filtering_level:
            body["filtering_level"] = filtering_level
        if video_nos:
            body["video_nos"] = list(video_nos)
        if tag:
            body["tag"] = tag
        if camera_tag:
            body["camera_tag"] = camera_tag
        if datetime_taken:
            body["datetime_taken"] = datetime_taken
        if latitude is not None:
            body["latitude"] = latitude
        if longitude is not None:
            body["longitude"] = longitude

        data = self._post_with_retry(
            f"{self.api_host}/search",
            headers=self._headers,
            json=body,
            timeout=self.timeout,
        )
        return data if isinstance(data, list) else []

    def search_audio_transcripts(
        self,
        query: str,
        *,
        unique_id: str = "default",
        page: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.api_host}/search_audio_transcripts",
            headers=self._headers,
            params={
                "query": query,
                "unique_id": unique_id,
                "page": page,
                "page_size": page_size,
            },
            timeout=self.timeout,
        )
        return self._unwrap(resp)

    def download_to_file(self, video_no: str, dest_path: str, *, unique_id: str = "default") -> str:
        """Stream the original video file to disk via POST /download.

        Returns the destination path. The endpoint streams binary on success
        and a JSON error envelope on failure — this method surfaces the
        latter as a MemoriesError.
        """
        with self.session.post(
            f"{self.api_host}/download",
            headers=self._headers,
            json={"video_no": video_no, "unique_id": unique_id},
            stream=True,
            timeout=self.timeout,
        ) as resp:
            if resp.status_code >= 400:
                raise MemoriesError(
                    f"HTTP {resp.status_code} from /download: {resp.text[:400]}",
                    payload=resp.text,
                )
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct.lower():
                body = resp.json()
                raise MemoriesError(
                    f"/download failed: code={body.get('code')} msg={body.get('msg')}",
                    code=body.get("code"),
                    payload=body,
                )
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        return dest_path

    def resolve_media_url(self, video_no: str) -> str:
        """Resolve a video_no to a public URL the VLM can fetch.

        /download streams the raw bytes — it does NOT return a hosted URL. To
        feed a video to /vu/chat/completions the customer must re-host the
        file on their own storage. Configure that bridge either via:

          - the ``MEMORIES_MEDIA_URL_TEMPLATE`` env var, e.g.
            ``https://my-cdn.example.com/{video_no}.mp4`` (``{video_no}`` is
            substituted at lookup time), or

          - an explicit map passed at MemoriesClient construction time
            (``media_url_map={video_no: url, ...}``).

        Raises MemoriesError if neither is configured. Agents that need to
        reason over media should treat this as a soft failure and fall back
        to a retrieval-only summary.
        """
        if self.media_url_map and video_no in self.media_url_map:
            return self.media_url_map[video_no]
        tpl = os.environ.get("MEMORIES_MEDIA_URL_TEMPLATE")
        if tpl:
            return tpl.format(video_no=video_no)
        raise MemoriesError(
            f"No media URL bridge configured for {video_no}. "
            "Set MEMORIES_MEDIA_URL_TEMPLATE or pass media_url_map. "
            "See README.md > 'Hosting videos for the VLM'."
        )

    # -------- Visual Intelligence: VLM completions ---------------------------

    def vlm_complete(
        self,
        prompt: str,
        *,
        video_url: str | None = None,
        image_url: str | None = None,
        mime_type: str = "video/mp4",
        system: str | None = None,
        model: str | None = None,
        response_json: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        strip_code_fences: bool = True,
    ) -> str:
        # /vu/chat/completions requires `content` to be an array even when there
        # is only a text part — sending a bare string is rejected with
        # "Model input cannot be empty". System messages still take a string.
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if video_url:
            content.append({"type": "input_file", "file_uri": video_url, "mime_type": mime_type})
        if image_url:
            content.append({"type": "input_file", "file_uri": image_url, "mime_type": "image/jpeg"})

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        request_body: dict[str, Any] = {
            "model": model or os.environ.get("MEMORIES_VLM_MODEL") or DEFAULT_VLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_json:
            request_body["extra_body"] = {"metadata": {"response_mime_type": "application/json"}}

        resp = self.session.post(
            f"{self.vlm_host}/vu/chat/completions",
            headers=self._headers,
            json=request_body,
            timeout=self.timeout * 2,
        )
        if resp.status_code >= 400:
            raise MemoriesError(
                f"HTTP {resp.status_code} from {resp.url}: {resp.text[:400]}",
                payload=resp.text,
            )
        body = resp.json()

        # The endpoint can return HTTP 200 with status="errored" — surface that
        # as a typed error rather than letting downstream parsing fail.
        if body.get("status") == "errored" or body.get("error"):
            err = body.get("error") or {}
            raise MemoriesError(
                f"VLM error: {err.get('code')} {err.get('message')}",
                code=err.get("code"),
                payload=body,
            )
        if "choices" not in body or not body["choices"]:
            raise MemoriesError(f"VLM returned no choices: {body}", payload=body)

        choice = body["choices"][0]
        # Two response shapes observed in the wild:
        #   - Gemini-style: {"text": "..."} per choice
        #   - OpenAI-style: {"message": {"content": "..."}}
        text = choice.get("text")
        if text is None:
            msg = choice.get("message") or {}
            text = msg.get("content", "")

        if strip_code_fences:
            text = _strip_json_code_fence(text)
        return text


def _strip_json_code_fence(text: str) -> str:
    """Gemini often wraps JSON output in ```json ... ``` fences even when
    response_mime_type=application/json. Trim them so callers can json.loads()
    the result directly.
    """
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3].rstrip()
    return s
