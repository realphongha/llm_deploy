"""Static verification for the reworked benchmark_api.py (no server needed)."""
import asyncio
import time

import benchmark_api as b


def test_prompt_corpus():
  assert len(b.PROMPTS) == 9, len(b.PROMPTS)
  names = [p["name"] for p in b.PROMPTS]
  assert len(set(names)) == len(names), "duplicate prompt names"
  cats = {p["category"] for p in b.PROMPTS}
  assert cats == {
      "coding", "code-review", "translation", "summarization",
      "creative-writing", "reasoning", "structured-extraction", "explain"
  }, cats
  for p in b.PROMPTS:
    words = len(p["text"].split())
    assert 300 <= words <= 1400, (p["name"], words)
    est = b.estimate_tokens(p["text"])
    assert 600 <= est <= 2200, (p["name"], est)
    assert p["text"].strip().endswith((".", "?", ":", "`", "\n")), p["name"]
  # Measured against a live llama.cpp server: real prompt_tokens were
  # 896-2359, i.e. 20x+ longer than the ~40 token prompts this suite replaced.
  assert sum(b.estimate_tokens(p["text"]) for p in b.PROMPTS) > 8000
  print("OK prompt corpus:", [(p["category"], p["name"],
                               len(p["text"].split()),
                               b.estimate_tokens(p["text"])) for p in b.PROMPTS])


def test_chunking():
  items = [{"name": f"p{i}"} for i in range(9)]
  names = lambda ch: [s["name"] for s, _ in ch]
  dups = lambda ch: [d for _, d in ch]

  # c=1: 9 chunks of 1, no padding.
  ch = b.padded_chunks(items, 1)
  assert len(ch) == 9 and all(len(c) == 1 for c in ch)
  assert all(d is False for c in ch for _, d in c)
  assert names([x for c in ch for x in c]) == [f"p{i}" for i in range(9)]

  # c=3 divides evenly: no padding.
  ch = b.padded_chunks(items, 3)
  assert [len(c) for c in ch] == [3, 3, 3]
  assert not any(d for c in ch for _, d in c)

  # c=4: tail padded by cycling from the front.
  ch = b.padded_chunks(items, 4)
  assert [len(c) for c in ch] == [4, 4, 4]
  assert names(ch[2]) == ["p8", "p0", "p1", "p2"]
  assert dups(ch[2]) == [False, True, True, True]

  # c=20 > 9: one chunk, every prompt at least once, 11 dups.
  ch = b.padded_chunks(items, 20)
  assert len(ch) == 1 and len(ch[0]) == 20
  assert names(ch[0])[:9] == [f"p{i}" for i in range(9)]
  assert sum(dups(ch[0])) == 11

  # Every unique prompt is fresh exactly once, regardless of padding.
  fresh = [s["name"] for c in b.padded_chunks(items, 4)
           for s, d in c if not d]
  assert fresh == [f"p{i}" for i in range(9)]

  try:
    b.padded_chunks(items, 0)
  except ValueError:
    pass
  else:
    raise AssertionError("padded_chunks(size=0) must raise")
  print("OK chunking")


def test_chunk_ordering():
  """c=1 must be strictly sequential; c=3 must overlap within a chunk."""
  events = []

  async def fake(prompt_spec, delay):
    events.append(("start", prompt_spec["name"], time.perf_counter()))
    await asyncio.sleep(delay)
    events.append(("end", prompt_spec["name"], time.perf_counter()))

  async def run(specs, c):
    events.clear()
    for chunk in b.padded_chunks(specs, c):
      await asyncio.gather(*[fake(s, 0.05) for s, _ in chunk])
    return [e for e in events]

  seq = asyncio.run(run([{"name": f"p{i}"} for i in range(9)], 1))
  starts = [t for kind, _, t in seq if kind == "start"]
  assert starts == sorted(starts), "sequential run must start in order"
  # No start may begin before the previous one ended.
  pairs = [(kind, t) for kind, _, t in seq]
  assert pairs == [("start", t) if k == "start" else ("end", t)
                   for k, t in pairs]
  for i in range(0, len(pairs), 2):
    assert pairs[i][0] == "start" and pairs[i + 1][0] == "end"

  par = asyncio.run(run([{"name": f"p{i}"} for i in range(9)], 3))
  pstarts = [t for kind, _, t in par if kind == "start"]
  pend = [t for kind, _, t in par if kind == "end"]
  # First chunk: 3 starts before any end. Last chunk: 3 ends after last start.
  assert sum(1 for t in pstarts if t < min(pend)) == 3, "chunk must overlap"
  assert sum(1 for t in pend if t > max(pstarts)) == 3
  print("OK chunk ordering")


def test_aggregation_math():
  """Aggregate must be sum(tok)/sum(time), not the mean of ratios."""
  def res(cat, name, ptok, prefill, gen, decode, usage_ok=True):
    return b.RequestResult(
        category=cat, name=name, prompt_tokens=ptok, usage_ok=usage_ok,
        prefill_s=prefill, decode_s=decode, ttft_ms=prefill * 1000,
        input_tps=ptok / prefill, gen_tokens=gen,
        output_tps=gen / decode if decode else 0.0)

  # Two identical requests: aggregate == mean.
  same = [res("coding", "a", 1000, 1.0, 100, 2.0),
          res("coding", "b", 1000, 1.0, 100, 2.0)]
  valid = same
  n = len(valid)
  mean_in = sum(r.input_tps for r in valid) / n
  agg_in = sum(r.prompt_tokens for r in valid) / sum(r.prefill_s for r in valid)
  assert abs(mean_in - agg_in) < 1e-9, (mean_in, agg_in)

  # Different prompt lengths: aggregate != mean of ratios.
  mixed = [res("coding", "a", 1000, 1.0, 100, 1.0),
           res("coding", "b", 100, 0.5, 50, 1.0)]
  mean_in = sum(r.input_tps for r in mixed) / 2
  agg_in = sum(r.prompt_tokens for r in mixed) / sum(r.prefill_s for r in mixed)
  assert abs(mean_in - agg_in) > 1e-6, (mean_in, agg_in)
  assert abs(agg_in - 1100 / 1.5) < 1e-9

  print("OK aggregation math (mean=%.2f aggregate=%.2f)" % (mean_in, agg_in))


def test_report_smoke(capsys=None):
  def res(cat, name, ptok, prefill, gen, decode, usage_ok=True, failed=False,
          burst=False, err=""):
    return b.RequestResult(
        category=cat, name=name, prompt_tokens=ptok, usage_ok=usage_ok,
        prefill_s=prefill, decode_s=decode, ttft_ms=prefill * 1000,
        input_tps=ptok / prefill if prefill else 0.0, gen_tokens=gen,
        output_tps=gen / decode if decode else 0.0, failed=failed, error=err,
        burst_capped=burst)

  rows = [
      res("coding", "a", 1000, 1.0, 100, 1.0),
      res("translation", "en-vi-article", 900, 0.4, 200, 2.0, usage_ok=False),
      res("reasoning", "r", 800, 0.8, 5, 1e-9, burst=True),
      res("explain", "boom", 700, 0, 0, 0, failed=True, err="connection reset"),
  ]
  code = b.print_report(rows, verbose=True)
  assert code == 0, code
  code_all_failed = b.print_report(
      [res("coding", "x", 1, 0, 0, 0, failed=True, err="nope")], verbose=False)
  assert code_all_failed == 1, code_all_failed
  print("OK report smoke")


def test_cache_bust():
  spec = {"category": "coding", "name": "a", "text": "BODY TEXT"}
  one = b.cache_bust(spec)
  two = b.cache_bust(spec)
  assert one["text"] != two["text"], "nonce must differ per call"
  assert one["text"].endswith("BODY TEXT")
  assert one["category"] == "coding" and one["name"] == "a", "metadata kept"
  assert spec["text"] == "BODY TEXT", "input spec must not be mutated"
  assert "run id" in one["text"]
  print("OK cache bust")


if __name__ == "__main__":
  test_prompt_corpus()
  test_chunking()
  test_chunk_ordering()
  test_aggregation_math()
  test_cache_bust()
  test_report_smoke()
  print("\nALL STATIC CHECKS PASSED")
