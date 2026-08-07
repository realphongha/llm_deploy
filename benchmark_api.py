import argparse
import asyncio
import sys
import time
from openai import AsyncOpenAI

# 10 Diverse benchmark prompts demanding detailed responses
TEST_PROMPTS = [
    (
        "Explain the fundamental differences between TCP and UDP protocols,"
        " including header structures, reliability mechanisms, and ideal use"
        " cases for each."
    ),
    (
        "Describe how Garbage Collection works in Java or Go, focusing on"
        " Mark-and-Sweep, Generational hypotheses, and stop-the-world pauses."
    ),
    (
        "How does the Raft consensus algorithm handle leader election and log"
        " replication? Explain step-by-step."
    ),
    (
        "Explain the Transformer architecture's self-attention mechanism using"
        " Query, Key, and Value matrices mathematically and conceptually."
    ),
    (
        "What is the CAP Theorem in distributed systems? Discuss how DynamoDB"
        " and PostgreSQL handle trade-offs between Consistency, Availability,"
        " and Partition Tolerance."
    ),
    (
        "Explain how Linux Process Scheduling (CFS - Completely Fair Scheduler)"
        " works using red-black trees and virtual runtime."
    ),
    (
        "Compare REST, gRPC, and GraphQL in terms of transport protocol,"
        " serialization format, performance, and schema definition."
    ),
    (
        "Walk me through what happens under the hood in the operating system"
        " when a page fault occurs during memory allocation."
    ),
    (
        "Explain the B-Tree and B+ Tree data structures and why B+ Trees are"
        " predominantly used for database indexing over binary search trees."
    ),
    (
        "How do modern CPUs execute speculative execution and branch"
        " prediction, and what security vulnerabilities (e.g., Spectre) do"
        " they introduce?"
    ),
]


async def fetch_first_model(client: AsyncOpenAI) -> str:
  """Queries /v1/models endpoint and returns the first available model ID."""
  try:
    models_page = await client.models.list()
    # The SDK returns a SyncPage/AsyncPage of Model objects
    models_list = models_page.data
    if not models_list:
      raise ValueError("Endpoint returned an empty list of models.")

    selected_model = models_list[0].id
    print(f"ℹ️ No --model provided. Auto-selected first model: '{selected_model}'")
    return selected_model
  except Exception as e:
    print(f"❌ Failed to fetch available models from /v1/models: {e}")
    sys.exit(1)


async def benchmark_single_request(
    client: AsyncOpenAI, prompt: str, model: str, max_tokens: int = 256
):
  """Executes a single async streaming request.

  Metrics:
    TTFT       : time from request start until first content token arrives.
    Input TPS  : prompt_tokens / prefill_time. prefill_time == TTFT (in
                 streaming, prefill happens before the first token emits).
    Output TPS : gen_tokens / decode_time. decode_time is the wall-clock span
                 from the first content token to the last content token, so
                 it measures actual generation throughput.
  """
  t_start = time.perf_counter()
  t_first = None      # arrival time of first content token
  t_last = None       # arrival time of last content token
  gen_tokens = 0      # content tokens counted from stream
  prompt_tokens = len(prompt.split())  # Fallback if server omits usage

  try:
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_tokens,
    )

    async for chunk in stream:
      # Token stream delta. Reasoning models stream thought tokens before
      # emitting visible content, and this also holds for llama.cpp servers
      # enabled with thinking, which emit them in a `reasoning_content`
      # field (rather than vLLM-style `reasoning`). Capture all of them so
      # the decode span isn't collapsed to a final burst of content tokens.
      if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        delta_text = (
            (delta.content or "")
            + (getattr(delta, "reasoning_content", "") or "")
            + (getattr(delta, "reasoning", "") or "")
        )
        if delta_text:
          now = time.perf_counter()
          if t_first is None:
            t_first = now
          t_last = now
          gen_tokens += 1

      # Final empty-choices chunk carries authoritative usage counts.
      if hasattr(chunk, "usage") and chunk.usage:
        if chunk.usage.prompt_tokens:
          prompt_tokens = chunk.usage.prompt_tokens
        if chunk.usage.completion_tokens:
          gen_tokens = chunk.usage.completion_tokens

    t_end = time.perf_counter()

    # Prefer stream-derived counts; fall back to last content arrival time.
    if t_last is None:
      t_last = t_end
    if t_first is None:
      t_first = t_end

    # 1) TTFT: request start -> first content token.
    ttft_ms = (t_first - t_start) * 1000

    # 2) Input (prefill) throughput: prompt tokens over prefill time (= TTFT).
    prefill_s = max(t_first - t_start, 1e-9)
    input_tps = prompt_tokens / prefill_s if prompt_tokens > 0 else 0.0

    # 3) Output (decode) throughput: generated tokens over decode span.
    # decode_time = time from first content token to last content token.
    # If the server returns everything in one burst, decode_time ≈ 0 and
    # output_tps spikes — that reflects your server's batching, not the
    # measurement. Use a tiny floor only to avoid division by zero.
    decode_s = max(t_last - t_first, 1e-9)
    output_tps = gen_tokens / decode_s if gen_tokens > 0 else 0.0

    return ttft_ms, input_tps, output_tps

  except Exception as e:
    print(f"\n[ERROR] Request failed: {e}")
    return 0.0, 0.0, 0.0


async def main():
  parser = argparse.ArgumentParser(
      description="Lightweight LLM TPS Benchmarker"
  )
  parser.add_argument(
      "--url",
      type=str,
      default="http://localhost:8002/v1",
      help="Base URL of OpenAI compatible server",
  )
  parser.add_argument(
      "--model",
      type=str,
      default=None,
      help="Target model name (Auto-fetches from /v1/models if omitted)",
  )
  parser.add_argument(
      "--key", type=str, default="dummy-key", help="API Key (default: dummy-key)"
  )
  parser.add_argument(
      "--concurrency", "-c",
      type=int,
      default=1,
      help="Number of concurrent streams to fire",
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=256,
      help="Max generation tokens per prompt",
  )
  args = parser.parse_args()

  client = AsyncOpenAI(base_url=args.url, api_key=args.key)

  # Auto-resolve model if not explicitly provided
  model_name = args.model
  if not model_name:
    model_name = await fetch_first_model(client)

  print(f"\n🚀 Starting Benchmark against: {args.url}")
  print(
      f"📊 Model: {model_name} | Concurrency: {args.concurrency} | Max Tokens:"
      f" {args.max_tokens}\n"
  )

  active_prompts = [
      TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(args.concurrency)
  ]

  tasks = [
      benchmark_single_request(client, prompt, model_name, args.max_tokens)
      for prompt in active_prompts
  ]

  results = await asyncio.gather(*tasks)

  valid_results = [r for r in results if r != (0.0, 0.0, 0.0)]
  if not valid_results:
    print("❌ All benchmark requests failed!")
    return

  avg_ttft = sum(r[0] for r in valid_results) / len(valid_results)
  avg_in_tps = sum(r[1] for r in valid_results) / len(valid_results)
  avg_out_tps = sum(r[2] for r in valid_results) / len(valid_results)

  print("=" * 50)
  print("📈 BENCHMARK RESULTS SUMMARY")
  print("=" * 50)
  print(f"⚡ Avg TTFT        : {avg_ttft:.2f} ms")
  print(f"📥 Avg Input TPS   : {avg_in_tps:.2f} tok/s")
  print(f"📤 Avg Output TPS  : {avg_out_tps:.2f} tok/s")
  print("=" * 50)


if __name__ == "__main__":
  asyncio.run(main())
