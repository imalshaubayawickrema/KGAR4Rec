import requests
import time
import os
from json import JSONDecodeError

def api_request(messages, agent, args):
    """
    messages: list of {"role": "system" | "user" | "assistant", "content": str}
    args: argparse args with model, api_key, temperature, max_retry_num
    """
    return gpt_api(messages, agent, args)


class ContentFilterError(Exception):
    """Raised when Azure content filter blocks a prompt. Retrying will not help."""
    pass


def gpt_api(messages, agent, args):

    url = args.url
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }


    max_retry_num = getattr(args, "max_retry_num", 5)
    retry_delay = 2

    while max_retry_num > 0:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as e:
            print(f"[warning] HTTP request error: {repr(e)}")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 3, 120)
            max_retry_num -= 1
            continue

        # Try to parse JSON safely
        try:
            data = resp.json()
        except JSONDecodeError:
            print("[warning] Non-JSON response:", repr(resp.text[:300]))
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 3, 120)
            max_retry_num -= 1
            continue

        # Success path
        if resp.status_code == 200 and "choices" in data:
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content")  # safe get

            if content is None:
                finish_reason = choice.get("finish_reason", "")
                print(f"[warning] content is None, finish_reason={finish_reason}")
                if finish_reason == "content_filter":
                    raise ContentFilterError("Azure content filter triggered on output.")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 3, 120)
                max_retry_num -= 1
                continue
            raw_usage = data.get("usage", {})
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                "completion_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }
            return content.strip(), usage

        # Error path
        err = data.get("error", {})
        err_code = err.get("code", "")
        inner = err.get("innererror", {})
        inner_code = inner.get("code", "")
        err_type = err.get("type") or data.get("type", "")
        err_msg = err.get("message") or data.get("message", "Unknown error")

        if err_code == "content_filter" or inner_code == "ResponsibleAIPolicyViolation":
            filter_result = inner.get("content_filter_result", {})
            triggered = [k for k, v in filter_result.items() if v.get("filtered")]
            print(f"[content_filter] Blocked by Azure content filter. Triggered categories: {triggered}")
            raise ContentFilterError(
                f"Azure content filter triggered: {triggered}. "
                f"Full result: {filter_result}"
            )

        elif err_type == "too_many_requests" or resp.status_code == 429:
            wait = max(retry_delay, 60)  # always wait at least 60s for rate limits
            print(f"[warning] Rate limit hit, waiting {wait}s before retry...")
            time.sleep(wait)
            retry_delay = min(retry_delay * 2, 120)
            max_retry_num -= 1
            continue

        elif err_type == "service_tier_capacity_exceeded":
            print(f"[warning] Service tier capacity exceeded, waiting {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 3, 120)
            max_retry_num -= 1
            continue

        else:
            print(f"[warning] API error ({err_type}): {err_msg}")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 3, 120)
            max_retry_num -= 1
            continue

    print("[error] Exhausted retries in gpt_api; returning None")
    return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
