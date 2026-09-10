"""Pure helpers for turning one vLLM scheduler output into model features.

This module deliberately has no vLLM dependency.  The in-container
``sitecustomize`` hook imports it, while unit tests exercise the bookkeeping
without requiring CUDA or a vLLM installation.
"""


def extract_requests(
    num_scheduled_tokens,
    new_requests,
    cached_ids,
    cached_computed,
    cached_output_tokens,
    prompt_length_cache,
):
    """Return the exact before-step state of every scheduled text request.

    ``new_requests`` contains ``(id, prompt_length, computed_tokens)`` tuples.
    vLLM classifies all newly admitted requests as context requests.  Cached
    requests with zero output tokens are also context/prefill requests; the
    rest are decode requests.  Prefix caching is disabled in this experiment,
    but all raw counts are retained so the assumption can be audited.
    """
    computed_by_id = {}
    output_by_id = {}
    new_ids = set()
    for request_id, prompt_length, computed_tokens in new_requests:
        prompt_length_cache[request_id] = int(prompt_length)
        computed_by_id[request_id] = int(computed_tokens)
        output_by_id[request_id] = 0
        new_ids.add(request_id)

    for request_id, computed_tokens, output_tokens in zip(
        cached_ids, cached_computed, cached_output_tokens
    ):
        computed_by_id[request_id] = int(computed_tokens)
        output_by_id[request_id] = int(output_tokens)

    requests = []
    for request_id, scheduled_tokens in num_scheduled_tokens.items():
        scheduled_tokens = int(scheduled_tokens)
        computed_tokens = computed_by_id[request_id]
        prompt_length = prompt_length_cache[request_id]
        output_tokens = output_by_id[request_id]
        is_prefill = request_id in new_ids or output_tokens == 0
        prompt_remaining = max(prompt_length - computed_tokens, 0)
        prefill_tokens = min(scheduled_tokens, prompt_remaining) if is_prefill else 0
        decode_tokens = scheduled_tokens - prefill_tokens

        requests.append(
            {
                "id": request_id,
                "phase": "prefill" if is_prefill else "decode",
                "scheduled_tokens": scheduled_tokens,
                "computed_tokens": computed_tokens,
                "prompt_tokens": prompt_length,
                "output_tokens": output_tokens,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
            }
        )
    return requests


def aggregate_features(requests):
    """Compute the exact regressors used by the additive iteration law.

    K uses the pre-step computed-token count, matching the router's resident-KV
    convention.  For each prefill grant s beginning at prefix p, its attention
    work is s * (p + s/2).
    """
    decode_requests = [request for request in requests if request["phase"] == "decode"]
    prefill_requests = [request for request in requests if request["phase"] == "prefill"]
    return {
        "B_decode": len(decode_requests),
        "B_scheduled": len(requests),
        "K_decode": sum(request["computed_tokens"] for request in decode_requests),
        "S_prefill": sum(request["prefill_tokens"] for request in prefill_requests),
        "U_prefill": sum(
            request["prefill_tokens"]
            * (request["computed_tokens"] + request["prefill_tokens"] / 2.0)
            for request in prefill_requests
        ),
        "decode_scheduled_tokens": sum(
            request["decode_tokens"] for request in decode_requests
        ),
        "prefill_requests": len(prefill_requests),
    }


def build_step_record(step, start_ns, end_ns, requests, scheduler_state=None):
    """Build one self-contained, JSON-serializable internal iteration row."""
    features = aggregate_features(requests)
    if features["B_decode"] and features["prefill_requests"]:
        regime = "mixed"
    elif features["prefill_requests"]:
        regime = "pure_prefill"
    elif features["B_decode"]:
        regime = "pure_decode"
    else:
        regime = "empty"
    return {
        "schema_version": 1,
        "step": int(step),
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "engine_step_s": (int(end_ns) - int(start_ns)) / 1_000_000_000.0,
        "regime": regime,
        "total_scheduled_tokens": sum(
            request.get(
                "scheduled_tokens",
                request.get("prefill_tokens", 0) + request.get("decode_tokens", 0),
            )
            for request in requests
        ),
        **features,
        "scheduler": scheduler_state or {},
        "requests": requests,
    }
