"""Lightweight LLM TPS benchmarker for OpenAI-compatible servers.

Runs a diverse suite of hand-written long prompts (coding, code review,
translation, summarization, creative writing, reasoning, structured
extraction, explanation) and reports per-category, overall-mean and
aggregate throughput numbers.

Input prompts are deliberately long (roughly 900-1400 tokens each) so prefill
dominates time-to-first-token and Input TPS becomes a meaningful, stable
measurement instead of a number dominated by per-request overhead.

Two measurement traps this script handles explicitly:

* Short prompts make Input TPS meaningless, and identical repeated prompts get
  served from a server-side prefix cache, which makes Input TPS look ~10x
  better than reality. Prompts are therefore long, and --cache-bust (default
  on) prepends a unique nonce so every run measures a cold prefill.
* The mean of per-request ratios is not throughput when prompt lengths differ,
  so both the mean and the aggregate (total tokens / total time) are printed.

Usage:
    python benchmark_api.py --url http://127.0.0.1:8007/v1
    python benchmark_api.py --url http://127.0.0.1:8008/v1 --model smart -c 3
    python benchmark_api.py --list-prompts
"""

import argparse
import asyncio
import secrets
import sys
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

# Lowest --max-model-len used by any launcher in this repo
# (gemma4/vllm_26b_translator.sh). Used only for a pre-flight warning.
CONSERVATIVE_MAX_MODEL_LEN = 32768

# Rough English words -> tokens factor, used for the pre-flight context check
# and --list-prompts only. Real counts come from server usage when returned.
WORDS_TO_TOKENS = 1.3

# Sent before measurement so cold-server state does not skew the first prompt.
WARMUP_PROMPT = "Reply with exactly one word: ready"


# --------------------------------------------------------------------------
# Prompt corpus
#
# Each PROMPTS entry: {"category": str, "name": str, "text": str}.
# Prompt length comes from realistic payload material (code, articles,
# transcripts, dossiers, data tables), never from repeated filler.
# Adding a prompt is one list entry plus one constant above PROMPTS.
# --------------------------------------------------------------------------

# coding / implement-module: ~1050 input tokens
_CODING_IMPLEMENT_MODULE = """\
You are a senior Python engineer. Implement one small module for an internal
LLM serving gateway. Follow the contract exactly: names, types, defaults and
exception behavior must match what is written here, because the module is
consumed by code that is already written and cannot be changed.

Module goal: per-API-key admission control that combines a token-bucket rate
limiter with a circuit breaker, so a broken backend cannot pile up unbounded
in-flight requests.

Required public API (module gateway_admission.py):

    class RateLimiter:
        def __init__(self, rate: float, capacity: float,
                     clock: Callable[[], float] = time.monotonic) -> None: ...
        def allow(self, key: str, cost: float = 1.0) -> bool: ...
        def retry_after(self, key: str, cost: float = 1.0) -> float: ...

    class CircuitBreaker:
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

        def __init__(self, failure_threshold: int = 5,
                     recovery_seconds: float = 30.0,
                     half_open_max_calls: int = 2,
                     clock: Callable[[], float] = time.monotonic) -> None: ...
        @property
        def state(self) -> str: ...
        def allow_request(self) -> bool: ...
        def record_success(self) -> None: ...
        def record_failure(self) -> None: ...

    class RateLimitError(Exception):
        def __init__(self, retry_after: float) -> None: ...

    class BackendUnavailableError(Exception): ...

    class AdmissionController:
        def __init__(self, limiter: RateLimiter, breaker: CircuitBreaker) -> None: ...
        def acquire(self, key: str, cost: float = 1.0) -> None:
            # Raise RateLimitError(retry_after) if the bucket rejects, or
            # BackendUnavailableError() if the breaker is open.
            ...

Hard constraints:
1. Python 3.11+, standard library only. No third-party imports.
2. No threads, locks, or asyncio: the gateway calls these objects from a single
   event loop thread, so keep the code free of reentrancy concerns.
3. Never mutate a caller's argument. No global or module-level mutable state:
   all per-key state lives inside the RateLimiter instance.
4. Buckets are created lazily per key on first use and must never store a
   token value above `capacity`.
5. Refill is lazy: computed from elapsed time between calls
   (tokens += elapsed * rate, clamped to capacity). Schedule nothing.
6. retry_after must return the exact seconds until `cost` tokens would be
   available using the same lazy refill, and 0.0 when the request would
   already be allowed.
7. Circuit breaker transitions: closed -> open when consecutive failures reach
   failure_threshold; while open, allow_request returns False until
   recovery_seconds elapse; then half_open admits at most half_open_max_calls
   probes; the first probe failure re-opens it immediately; a recorded success
   in half_open closes it and resets all counters.
8. record_success in the closed state resets the consecutive-failure count.
9. The injected clock callable is the only source of time. The unit tests pass
   a fake clock and advance it explicitly.
10. Keep the module under 200 lines. Docstrings on public classes and methods
    only. Raise ValueError from __init__ when any numeric parameter is
    non-positive or non-finite.
11. Key handling: allow() with an unknown key must create that bucket at full
    capacity and then immediately charge cost to it. Do not pre-populate keys
    and do not provide a way to enumerate them; keys carry customer identity and
    this module is loaded in a process that dumps memory on crash reports.
12. Do not use dataclasses, attrs, or TypedDict: the target deployment pins an
    interpreter build where those imports are forbidden by policy. Plain classes
    with __slots__ are allowed and preferred.
13. Floating point: bucket values drift when time deltas accumulate. Recompute
    from (last_seen, current) every call rather than integrating a running
    residual, and clamp to [0, capacity] on every write so a caller passing a
    huge cost can never poison a bucket into a permanently negative state.
14. Every public method must be safe to call 10000 times per second on the same
    key without allocating more than a couple of short-lived objects.

Acceptance tests your implementation must satisfy (check them silently, then
return code that passes all of them):

    limiter = RateLimiter(rate=1.0, capacity=3.0, clock=fake.now)
    assert limiter.allow("a") and limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert 0.9 < limiter.retry_after("a") <= 1.0
    fake.advance(1.0)
    assert limiter.allow("a")

    limiter.allow("b", cost=3.0)
    assert not limiter.allow("b", cost=0.5)
    assert limiter.retry_after("b", cost=3.0) > 2.9
    fake.advance(10.0)
    assert limiter.retry_after("b", cost=3.0) == 0.0
    assert limiter.allow("b", cost=3.0)

    # A huge cost must not poison the bucket permanently.
    limiter.allow("c", cost=1000.0)
    fake.advance(4.0)
    assert limiter.retry_after("c", cost=1.0) == 0.0

    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10.0,
                             half_open_max_calls=1, clock=fake.now)
    breaker.record_failure(); assert breaker.state == "closed"
    breaker.record_failure(); assert breaker.state == "open"
    assert not breaker.allow_request()
    fake.advance(10.0)
    assert breaker.state == "half_open" and breaker.allow_request()
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow_request()

Output format: return exactly one fenced Python code block containing the
complete module (imports first, then exceptions, then classes), with no prose
before or after the block. Explain nothing. Think through the lazy refill and
half-open edge cases before writing the final code.
"""

# code-review / review-refactor: ~1250 input tokens (target module included)
_CODE_REVIEW_TARGET = """\
# billing_cache.py  (internal, Python 3.11)
import time, os, json

CACHE = {}
FAILED = []
_last_flush = 0


def load_prices(path):
    global CACHE
    f = open(path)
    try:
        data = json.load(f)
    except:
        data = {}
        print("load failed")
    f.close()
    CACHE = data
    return CACHE


def get_price(sku, fallback=0):
    if sku in CACHE == False:
        load_prices(os.environ["PRICE_FILE"])
    if CACHE.get(sku) is None:
        return fallback
    return CACHE[sku]


def total(cart, region):
    s = 0
    taxes = []
    for item in cart:
        p = get_price(item["sku"])
        q = item.get("qty", 1)
        s = s + p * q
        for t in REGIONS.get(region, []):
            taxes.append(t)
    rate = 0
    for t in taxes:
        rate = rate + t
    return s * (1 + rate)


def discount(user, total):
    if user.is_vip == True:
        return total * 0.9
    if user.orders > 10:
        return total * 0.95
    return total


def flush(path):
    global _last_flush
    if time.time() - _last_flush < 60:
        return
    for k in CACHE:
        try:
            v = CACHE[k] * 1.1
            json.dumps({"k": k, "v": v})
        except:
            FAILED.append(k)
    with open(path, "w") as f:
        f.write(json.dumps(CACHE))
    _last_flush = time.time()


def report(rows):
    out = ""
    for r in rows:
        out = out + str(r["sku"]) + "," + str(r["price"]) + "\\n"
    open("report.csv", "w").write(out)
    return out


def is_wholesale(user):
    try:
        return user.tier == "wholesale" or user.orders >= 50
    except Exception:
        return False


def retry_load(path, tries=3):
    for i in range(tries):
        try:
            return load_prices(path)
        except Exception as e:
            print(e)
            time.sleep(2)


def update(sku, price, changes=[]):
    CACHE[sku] = price
    changes.append((sku, price))
    CACHE["__audit__"] = changes
    return changes


def summary(users):
    result = []
    for u in users:
        if is_wholesale(u):
            result.append(u)
    for u in result:
        if u.orders > 100:
            print("big", u)
    return len(result)
"""

_CODE_REVIEW_REFACTOR = (
    "You are the tech lead reviewing the module below. It ships in a billing"
    " path, so correctness and auditability matter more than brevity.\n\n"
    "Review checklist you must apply (judge which items are actually violated;"
    " do not invent violations):\n"
    "A. Correctness and logic errors, including comparison and precedence"
    " bugs.\n"
    "B. Mutation, aliasing and shared global state problems; concurrency safety"
    " if this module were imported by a multi-worker server process.\n"
    "C. Resource handling: file handles, error paths, partial writes.\n"
    "D. Exception handling: bare excepts, swallowed errors, missing context,"
    " exceptions used for control flow.\n"
    "E. Performance: needless repeated work, redundant or quadratic iteration,"
    " string building, cache reload storms.\n"
    "F. Configuration and environment handling, including missing keys and"
    " default values that silently produce wrong money amounts.\n"
    "G. Naming, dead code, unused state, and readability problems that will"
    " hurt the next engineer on call at 3am.\n"
    "H. Testability: what is hard to test today and how to make it testable"
    " without changing public behaviour.\n\n"
    "Deliverables, in this exact order:\n"
    "1. A findings table with columns: id, severity (blocker/major/minor),"
    " line or function, category letter (A-H), and a one-sentence description"
    " of the concrete impact. Order by severity. Find at least 12 real"
    " findings and quote the offending expression in each one.\n"
    "2. A fenced code block with the refactored module. Preserve the"
    " externally visible behaviour of load_prices, get_price, total, discount"
    " and report (same arguments, same return types) while fixing every"
    " blocker and major finding. Add type hints, replace globals with a small"
    " immutable config object or explicit parameters, and inject the clock and"
    " file paths where that improves testability. No third-party"
    " dependencies.\n"
    "3. A bullet list (max 6 bullets) of follow-up work you deliberately did"
    " not do, and why.\n\n"
    "Rules: Python 3.11 standard library only. Where two fixes conflict,"
    " state the tradeoff in one line. Do not summarise the code back to me;"
    " produce only findings, code, and follow-ups.\n\nModule under"
    " review:\n\n```python\n" + _CODE_REVIEW_TARGET + "```\n"
)

# translation / en-vi-article: ~1150 input tokens
_TRANSLATION_EN_VI = """\
Translate the article below from English into Vietnamese.

Requirements:
- Produce fluent, publication-quality Vietnamese in a formal technical blog
  register. Do not translate word by word; restructure sentences so a native
  Vietnamese engineer reads it naturally.
- Keep every Markdown heading level and every list nesting exactly as in the
  source.
- Do NOT translate: model names, product names, file names, CLI flags,
  environment variables, code identifiers, metric names, or numbers with
  units (for example `--max-model-len`, `tok/s`, NVFP4, KV cache).
- Use one consistent Vietnamese rendering for each term in the Glossary, and
  keep it consistent everywhere that term appears.
- Preserve the approximate length of the article. Do not summarise, compress,
  add commentary, or drop any paragraph.
- After the translation, output a two-column Markdown glossary table of the 12
  terms you were least sure about, with the columns: English, Vietnamese, Note
  (one short clause explaining why you chose that rendering).

Output structure:
## Translation
<the full Vietnamese article>

## Glossary
<table with 12 rows>

---
Article to translate:

# Squeezing a 27B Model onto a Desktop Inference Box

## Why the numbers on the model card are not your numbers

Model card throughput figures are almost always produced under conditions that
a small private deployment cannot reproduce. They are measured with a large
batch of identical prompts, a fixed output length, a warm cache, a single
precision format, and a serving stack tuned specifically for that one shape of
traffic. Your traffic is none of those things. It arrives in bursts, it mixes
five hundred token code completions with eight thousand token document
questions, and it expects the first token to appear before the developer
forgets what they were asking.

The practical consequence is that capacity planning for a private deployment is
a measurement problem, not a download problem. Before arguing about which
quantisation format to use, you need a benchmark that produces the three
numbers that actually decide user experience: prefill throughput, decode
throughput, and time to first token.

## Prefill and decode are different workloads wearing one process

The two phases of inference have opposite personalities. Prefill is
compute-bound: the model reads the entire prompt in parallel, and the amount of
floating point work grows with the number of prompt tokens. Decode is
memory-bound: every new token requires reading the weights and the whole key
value cache once more, and the arithmetic intensity is tiny. A GPU that is
saturated during prefill can be sitting almost idle during decode, and the only
way to lift decode utilisation is to run several requests at once.

This is why measuring prefill with a forty token prompt is useless. Forty tokens
take so little time that the number you report is dominated by HTTP handler
overhead, tokenisation, scheduler latency, and the occasional network hiccup.
To observe real prefill throughput you need prompts long enough for the matrix
multiplications to dominate the measurement: one to several thousand tokens,
ideally spanning the range your product actually serves.

## Quantisation is a trade, not a free lunch

Weight only quantisation to a four bit format reduces the data that must move
between memory and compute, which helps the memory-bound decode phase a great
deal and the compute-bound prefill phase much less. Full FP8 formats can help
prefill on hardware with fast low precision tensor cores, but they are more
sensitive to outliers and can degrade quality on tasks that require precise
instruction following, long chain reasoning, or exact JSON output.

A sane approach is to benchmark the same prompt suite against two or three
candidate formats and read the per category results rather than the average. An
average hides the fact that a format which is invisible on summarisation can
quietly destroy structured extraction, where one wrong bracket makes the whole
response unusable.

## The key value cache is the real memory budget

The key value cache, not the weights, usually decides how many requests your
server can hold concurrently. Cache size grows linearly with both context
length and the number of concurrent sequences, so a deployment that advertises
a large context window and high concurrency at the same time may be quietly
thrashing: admitting a request, then evicting it before it finishes generating.

Quantised key value caches buy back a surprising amount of headroom at a modest
quality cost, and prefix caching can make a repeated system prompt nearly free
in prefill. Both are worth measuring explicitly rather than trusting the
release notes.

## What to do on Monday

Build a small prompt suite that reflects your traffic: long document questions,
code generation, translation pairs, structured extraction, and open ended
writing. Run it sequentially to get clean per request numbers, then run it with
a small amount of concurrency to see what the scheduler actually delivers.
Compare aggregate throughput with the mean of the per request ratios. When
those two numbers disagree, the disagreement is telling you something real
about your workload distribution, and it is almost always worth listening to.
"""

# translation / vi-en-article: ~1000 input tokens (Vietnamese tokenises denser)
_TRANSLATION_VI_EN = """\
Dịch bài viết dưới đây từ tiếng Việt sang tiếng Anh.

Yêu cầu:
- Văn phong tiếng Anh trang trọng, tự nhiên, đúng thể loại blog kỹ thuật.
  Không dịch word-by-word; hãy tái cấu trúc câu sao cho người đọc bản ngữ thấy
  mạch lạc, nhưng tuyệt đối không thêm bình luận hay ý mới.
- Giữ nguyên cấp heading Markdown, thứ tự đoạn và mức lồng nhau của danh sách.
- Không dịch: tên mô hình, tên sản phẩm, tên file, tham số dòng lệnh, biến môi
  trường, định danh code, tên chỉ số, và các số kèm đơn vị (ví dụ
  `--max-model-len`, `tok/s`, NVFP4, DGX Spark).
- Dùng nhất quán một cách dịch cho mỗi thuật ngữ trong bảng Glossary.
- Giữ độ dài xấp xỉ bản gốc; không tóm tắt, không bỏ đoạn, không bịa số liệu.
- Sau bản dịch, xuất một bảng Markdown hai cột gồm 12 thuật ngữ bạn thấy khó và
  quan trọng nhất, với các cột: Vietnamese, English, Note (một mệnh đề ngắn giải
  thích lý do chọn cách dịch đó).

Cấu trúc đầu ra:
## Translation
<bản dịch tiếng Anh đầy đủ>

## Glossary
<bảng 12 dòng>

---
Bài viết cần dịch:

# Chạy mô hình 27B trên một hộp máy bàn: những con số thật

## Số của nhà sản xuất không phải số của bạn

Các con số throughput in trên trang mô hình hầu như luôn được đo trong điều
kiện mà một hệ thống triển khai nội bộ không thể tái hiện: batch lớn gồm các
prompt giống hệt nhau, độ dài đầu ra cố định, cache đã nóng, một định dạng lượng
hoá duy nhất, và stack phục vụ được tinh chỉnh đúng cho dạng traffic đó. Traffic
của chúng ta thì ngược lại. Nó đến từng burst, trộn lẫn những yêu cầu viết code
năm trăm token với những câu hỏi tài liệu tám nghìn token, và người dùng thì mong
token đầu tiên xuất hiện trước khi họ kịp quên mình vừa hỏi gì.

Hệ quả thực tế là việc hoạch định năng lực cho một hệ thống nội bộ là bài toán đo
đạc, không phải bài toán tải mô hình về. Trước khi tranh luận nên chọn định dạng
lượng hoá nào, ta cần một bộ benchmark tạo ra ba con số quyết định trải nghiệm
người dùng: throughput prefill, throughput decode, và thời gian chờ token đầu
tiên.

## Prefill và decode là hai workload khác nhau trong cùng một tiến trình

Hai pha của suy luận có tính chất trái ngược nhau. Prefill chịu ràng buộc tính
toán: mô hình đọc toàn bộ prompt song song, lượng phép tính tăng theo số token
của prompt. Decode chịu ràng buộc bộ nhớ: mỗi token mới buộc ta đọc lại toàn bộ
trọng số và toàn bộ cache key-value một lần nữa, còn cường độ tính toán trên mỗi
byte thì rất nhỏ. Một GPU bão hoà khi prefill có thể gần như ngồi không khi
decode, và cách duy nhất để nâng mức sử dụng ở pha decode là chạy nhiều request
cùng lúc.

Đó là lý do việc đo prefill bằng prompt bốn mươi token là vô nghĩa. Bốn mươi
token tốn ít thời gian đến mức con số ta báo bị chi phối bởi overhead của HTTP
handler, bước token hoá, độ trễ scheduler và những cú trượt mạng ngẫu nhiên.
Muốn thấy throughput prefill thật, prompt phải đủ dài để các phép nhân ma trận
chiếm ưu thế: từ một đến vài nghìn token, lý tưởng là phủ đúng dải độ dài mà sản
phẩm phục vụ.

## Lượng hoá là một sự đánh đổi, không phải bữa trưa miễn phí

Lượng hoá trọng số xuống định dạng bốn bit làm giảm lượng dữ liệu phải di chuyển
giữa bộ nhớ và nhân tính, nên cải thiện nhiều cho pha decode và rất ít cho pha
prefill. Các định dạng FP8 đầy đủ có thể giúp prefill trên phần cứng có tensor
core tốc độ cao, nhưng nhạy cảm hơn với outlier và có thể làm giảm chất lượng ở
những tác vụ đòi hỏi tuân thủ chỉ dẫn chính xác, suy luận chuỗi dài, hoặc xuất
JSON đúng khuôn.

Cách làm hợp lý là chạy cùng một bộ prompt cho hai hoặc ba định dạng ứng viên và
đọc kết quả theo từng nhóm tác vụ thay vì đọc một con số trung bình. Số trung
bình che mất sự thật rằng một định dạng vô hại với tóm tắt văn bản lại có thể
phá hỏng âm thầm việc trích xuất có cấu trúc, nơi chỉ một dấu ngoặc sai cũng
khiến phản hồi trở nên vô dụng.

## Cache key-value mới là ngân sách bộ nhớ thật

Cache key-value, chứ không phải trọng số, thường quyết định số request mà server
giữ được đồng thời. Kích thước cache tăng tuyến tính theo cả độ dài ngữ cảnh lẫn
số sequence chạy song song, nên một hệ thống quảng cáo ngữ cảnh lớn và concurrency
cao cùng lúc rất có thể đang thrashing trong im lặng: nhận request rồi đẩy nó ra
trước khi kịp sinh xong token.

Cache key-value lượng hoá lấy lại một khoảng headroom đáng kể với chi phí chất
lượng chấp nhận được, còn prefix caching có thể khiến system prompt lặp lại gần
như miễn phí ở pha prefill. Cả hai đều đáng được đo đàng hoàng thay vì tin vào
release note.

## Việc nên làm vào thứ Hai

Hãy dựng một bộ prompt phản ánh đúng traffic của mình: câu hỏi tài liệu dài, sinh
code, cặp dịch song ngữ, trích xuất có cấu trúc và viết mở. Chạy tuần tự để lấy số
sạch cho từng request, sau đó chạy với một mức concurrency nhỏ để xem scheduler
thực sự giao hàng ra sao. So sánh throughput tổng hợp với trung bình của các tỉ số
trên từng request; khi hai con số này lệch nhau, chính độ lệch đó đang nói với ta
một điều rất thật về phân phối workload của hệ thống.
"""

# summarization / meeting-minutes: ~1100 input tokens
_SUMMARIZE_TRANSCRIPT = """\
You are the rotating scribe for a weekly engineering release meeting. Below is
a raw, lightly cleaned transcript. Produce structured meeting minutes.

Output exactly these sections, in this order, using the given headings:

## Decisions
Numbered list. Only decisions that were actually agreed. If a decision was
conditional, state the condition. Each item ends with the deciding person's name
in parentheses.

## Action items
A Markdown table with columns: Owner | Action | Due | Blocking?. Use the earliest
date mentioned for the Due column; if none was mentioned, write "unspecified".
Blocking? is yes/no and must reflect whether someone said another workstream
waits on it.

## Risks and open questions
Bullet list. Include anything flagged as risky, uncertain, or explicitly left
unresolved. Do not downgrade a stated risk into a neutral note.

## Summary
120-170 words, written for an executive who will not read the transcript. It must
mention the release date decision, the one item that is currently blocking, and
the budget constraint. No information beyond the transcript.

Rules: never attribute a statement to a person who did not say it; never merge
two separate decisions into one; keep numbers, dates and version strings exactly
as spoken; if two people contradict each other, record the contradiction as an
open question rather than picking a side.

TRANSCRIPT:

ANH: Okay, fourteen fifteen, let us start. Main goal today is the 3.6 release
train. Dung, do we still have the twelfth?
DUNG: We had the twelfth. After the quantisation regression on the extraction
task I would not ship on the twelfth. I need at least three more days of
benchmarking.
BINH: Three days puts us on the fifteenth, that is a Friday.
DUNG: Friday the fourteenth actually. Friday the fifteenth if we also wait for
the tokenizer patch.
MAI: Can we separate them? The tokenizer patch fixes the Vietnamese token
counting, it is not related to quantisation.
DUNG: Correct, they are separate. I would ship the quantisation hotfix on the
fourteenth and take tokenizer with it if it is ready, otherwise the following
Tuesday.
ANH: Decided then. Fourteenth for the release, tokenizer is a may-ride. Dung
owns the go or no-go. Next, the benchmark suite. Binh, status?
BINH: The suite runs but the prompts were forty words long, so prefill numbers
were noise. Output tokens per second looked fine, input tokens per second moved
thirty percent between runs. I rewrote nine prompts across coding, code review,
translation both directions, summarisation, creative writing, reasoning and
structured extraction. Each is around a thousand tokens.
MAI: Good, but who maintains nine long prompts? They drift.
BINH: They are checked in as data in one file. Adding a tenth prompt is one list
entry.
ANH: Fine. Binh lands the suite this week with per-category reporting and an
aggregate line, not only the mean.
BINH: Acknowledged. That one is not blocking.
CHI: I want to raise the concurrency question. Our dashboards report the mean of
per-request ratios. With long prompts that number will look worse than the
aggregate, because prefill time per request is much bigger now.
ANH: Then report both. If they disagree, we read the disagreement.
CHI: Agreed, but someone has to define which one is the SLO. I would say
aggregate for capacity, mean for user experience.
ANH: Mai, can your team own the SLO definition?
MAI: I can own the document, not before next Thursday. I need Chi to give me the
current p95 numbers first.
CHI: p95 of what, time to first token under concurrency three?
MAI: Yes, and decode throughput at the same point.
CHI: I can have that Monday.
DUNG: Back to quantisation. The four bit weights are fine on summarisation and
translation. On JSON extraction, two of the nine runs returned a trailing comma.
That is a fail.
ANH: Is it the checkpoint or the server?
DUNG: I believe the checkpoint. The FP8 candidate does not show it, but it is six
percent slower on decode, which is our main cost.
BINH: We could add a repair step in the gateway for trailing commas.
DUNG: No. That hides a model defect and we lose the signal. I would rather delay
than ship that.
ANH: Agreed, no silent repair. Dung, file it as a blocker against the four bit
candidate and record the reproduction runs so we can show the vendor.
DUNG: Filing today. Blocking, yes.
MAI: Budget. We are over on cloud inference this month because the eval jobs run
against the hosted API. I am asking to move evals onto the local boxes after
hours.
ANH: Approved in principle. Cost is a hard constraint this quarter, we are not
increasing the cloud line.
MAI: Then I need the second box. It sits idle at nights.
CHI: It is not idle. Nightly integration tests use it from eleven to two.
MAI: Evals can run two to seven. Is that enough?
CHI: Two to seven is fine. But if evals are still running at seven, the tests
have to win.
MAI: Fair, I will pre-empt evals at seven.
ANH: Decided. Mai and Chi write the shared schedule, no cloud increase. Mai also
drafts the SLO doc by Thursday.
BINH: One more. The vision server is not ready. Should the benchmark suite cover
image inputs?
ANH: Not this round. Text first, minimal version. Add image later when that
server is stable and we have a real requirement.
BINH: Noted. Also the translation model is a dedicated build, so I keep both
translation directions in the suite.
MAI: Please also keep a Vietnamese source document, not only English to
Vietnamese. Our traffic is mostly the other direction.
BINH: I have both.
DUNG: Reminder that the meeting length is the problem, not the suite.
ANH: Noted, and I am taking that as criticism. Anything else? No. Fourteen
twenty-nine, meeting closed.
"""

# creative-writing / scene-dossier: ~850 input tokens
_CREATIVE_WRITING_DOSSIER = """\
You are a fiction writer working on the opening chapter of a literary near-future
novel. Write one scene from the material below.

SETTING DOSSIER (treat as canon; do not contradict it)

The story takes place in Hai Duong, a fictional coastal Vietnamese city in the
present-plus decade, where the tide floods the old market street four or five
mornings a month and the city has stopped pretending this is temporary. The
municipality runs a small predictive system, locally called "the Almanac",
that forecasts which mornings will flood. It was built cheaply by a contractor
who left the country, and it is quietly, stubbornly good. Nobody can improve it.

Main character: Du, thirty-four, a hydrologist who took a municipal desk job
after her fieldwork funding collapsed. She is the person who maintains the
Almanac, which is a euphemism for: she listens to it. She keeps a paper ledger
in which she writes down, by hand, every morning the Almanac was wrong. The
ledger is on page eleven. It has been on page eleven for three years.

Second character: Thuy, nineteen, a delivery rider and part-time motorbike-taxi
driver who has never once been wrong about a flood because she simply refuses to
take the low road when the water smells like mud. Thuy thinks the Almanac is a
kind of god and Du is its bored priest.

Third presence: the flood itself, rendered as a physical thing with weight and
schedule, never as a metaphor announced in the text.

Supporting notes on place and tone

The control room is a converted customs office on the second floor of a market
building: three dead UPS units nobody replaced, a wall fan that turns the room's
paper into weather, a kettle with a hairline crack, and a window that looks out
over the market roof toward the water. Everything the Almanac says arrives as a
single printed line at 03:40, and Du has read that line eleven hundred mornings
in a row. Outside, the city is asleep in the specific way coastal cities are
asleep, with engines running because a flooded street can appear in eleven
minutes.

The novel's tone is dry and procedural. The interest is in the gap between an
institution's confidence and a person's evidence. Du is not a skeptic and not a
believer; she is a woman who has written eleven hundred correct results into a
ledger and who now has to explain to a dripping teenager why the ledger might be
right and the street wrong. Thuy is not comic relief; she is competent in a way
the municipality has never paid for, and she knows it.

Recurring objects you may use, exactly once each: a thermos, a plastic marker
cap, a rubber band of motorbike keys. Do not name the Almanac's vendor.

Context of the scene: it is four in the morning. The Almanac says no flood. Du,
alone in the control room, sees a reading that contradicts it for the first time
in three years. Thuy has just knocked on the door of the control room because
the water is already up to her front wheel and she has come to tell the person
she believes is in charge. Du has to decide, in the space of about five minutes,
whether to trust the machine that has never failed her or the girl who is wet,
and what she decides will cost her something either way.

Constraints (these are hard):
1. Write in English, close third person limited on Du, past tense. Keep the
   Vietnamese forms of address and the few Vietnamese words Thuy uses, with no
   italicising and no glossary.
2. Between 550 and 700 words. Do not exceed 700.
3. Do not use any of these words: devastating, unprecedented, breathtaking,
   sinister, whisper, dance, tapestry, testament, delve.
4. No backstory paragraphs. Any history you need must arrive inside action or
   dialogue, and no single paragraph may be longer than 90 words.
5. The scene must contain: one sound described precisely; one object Du touches
   twice for different reasons; Thuy speaking at least three times, in her own
   register, in sentences shorter than Du's; and one number stated out loud.
6. End on a physical action, not on a thought, and not on a reveal of the
   Almanac's prediction.
7. Do not resolve the decision. The chapter continues after this scene.
8. Prose style: concrete, dry, unsentimental; the emotion comes from what Du
   declines to say. Avoid rhetorical questions.

Before writing, plan silently: the exact order of beats, where Thuy's first line
lands, and how the second touch of the object pays off. Then output ONLY the
scene text, with no title, no plan, and no commentary.
"""

# reasoning / multi-step-analysis: ~950 input tokens
_REASONING_CAPACITY = """\
You are doing capacity planning for a private LLM inference fleet. Work the
problem below part by part. Show every intermediate number you use, state each
formula before applying it, and keep units visible in every line. Round money to
whole USD and throughput to one decimal.

Fleet and workload data

| Node | GPU | Model served      | Max len (tok) | Max concurrent seq | Monthly cost (USD) |
|------|-----|-------------------|---------------|--------------------|--------------------|
| n1   | A   | Qwen3.6-27B-NVFP4 | 65536         | 8                  | 1200               |
| n2   | A   | Qwen3.6-27B-NVFP4 | 65536         | 8                  | 1200               |
| n3   | B   | Gemma-4-26B-A4B   | 32768         | 16                 | 900                |
| n4   | C   | Qwen3.6-35B-A3B   | 32768         | 4                  | 1500               |

Measured single-request behaviour (sequential, warm server):

| Model               | Prefill (tok/s) | Decode (tok/s) |
|---------------------|-----------------|----------------|
| Qwen3.6-27B-NVFP4   | 2100            | 46             |
| Gemma-4-26B-A4B     | 3400            | 71             |
| Qwen3.6-35B-A3B     | 1600            | 88             |

Assume decode throughput stays flat up to the node's max concurrent sequences
and is not available beyond it (requests over the limit queue, they are not
served). Assume prefill throughput is shared: with k concurrent sequences the
effective prefill throughput per sequence is prefill_throughput divided by k.

Traffic profile (weekday daytime peak, steady state, 8 hours):

| Segment       | Share of requests | Avg input tok | Avg output tok |
|---------------|-------------------|---------------|----------------|
| Doc questions | 0.40              | 6000          | 300            |
| Coding        | 0.30              | 1500          | 900            |
| Translation   | 0.20              | 900           | 850            |
| Extraction    | 0.10              | 2200          | 120            |

Memory and cache data

| Model               | Weight mem (GiB) | KV cache per 1k tok (GiB) |
|---------------------|------------------|---------------------------|
| Qwen3.6-27B-NVFP4   | 15.0             | 0.9                       |
| Gemma-4-26B-A4B     | 14.0             | 1.4                       |
| Qwen3.6-35B-A3B     | 19.0             | 0.6                       |

Usable memory for weights plus KV cache per node: 72 GiB. Prefix caching is off
and must be treated as off for this calculation.

Peak arrival rate: 1800 requests in the 8 hour window, on the mixed profile
above. All requests are served by a single model choice of your selection.

Questions

Part 1. For each model, compute the blended per-request service demand: total
prefill seconds and total decode seconds per request, weighted by the segment
shares. Present a table with columns: Model | prefill s/req | decode s/req.

Part 2. Compute the maximum sustained requests per hour a single node can sustain
for your chosen model, using the concurrency-limited prefill rule, at concurrency
1, 2, 4 and at that node's max concurrent sequences. State which concurrency
maximises throughput and by how much it beats concurrency 1.

Part 3. Decide the minimum set of nodes needed to serve the peak without any
request waiting longer than 90 seconds for its first token, under the simple rule
that a request's wait equals the prefill time of the sequence whose prefill it
must share. Show the arithmetic that proves your set is sufficient and the
arithmetic that shows one fewer node is insufficient.

Part 4. Compute the monthly cost per million output tokens for your chosen
deployment, and for the cheapest alternative that also satisfies Part 3. Report
the difference in USD per million output tokens.

Part 5. Using the memory table, verify that your chosen node set can actually
hold the concurrency you assumed in Part 2 for the Doc questions segment at its
6000 token input plus 300 token output. Report KV cache GiB per sequence and per
node, and the maximum concurrency the memory allows. If memory caps concurrency
below the value you chose in Part 2, redo the throughput arithmetic and say so
explicitly.

Part 6. Sanity checks, mandatory. (a) Recompute total peak hourly demand two ways:
from per-request seconds summed over segments, and from 1800 requests times
blended seconds, and confirm they agree within one percent. (b) State one
assumption in this data set that is most likely to be wrong in reality, and
describe directionally how your node count would change if it were wrong.

Output format: sections titled Part 1 through Part 6. Under each, first the
formula in words, then the arithmetic, then the result in bold. End with a single
line: RECOMMENDATION: <model> on <nodes> at concurrency <k>, <$X per 1M output
tokens>.
"""

# structured-extraction / json-schema: ~1400 input tokens
_EXTRACTION_SCHEMA = """\
{
  "type": "object",
  "required": ["company", "funding", "deployments", "hiring", "financials",
               "regulatory", "outage", "product_launches", "market",
               "unconfirmed"],
  "additionalProperties": false,
  "properties": {
    "company": {
      "type": "object",
      "required": ["name", "hq_city", "country", "ceo"],
      "additionalProperties": false,
      "properties": {
        "name": {"type": "string"},
        "hq_city": {"type": "string"},
        "country": {"type": "string"},
        "ceo": {"type": "string"}
      }
    },
    "funding": {
      "type": "object",
      "required": ["round", "amount_usd_millions", "lead_investor",
                   "participating_investors",
                   "pre_money_valuation_usd_millions", "conditions"],
      "additionalProperties": false,
      "properties": {
        "round": {"type": "string"},
        "amount_usd_millions": {"type": ["number", "null"]},
        "lead_investor": {"type": ["string", "null"]},
        "participating_investors": {"type": "array", "items": {"type": "string"}},
        "pre_money_valuation_usd_millions": {"type": ["number", "null"]},
        "conditions": {"type": "array", "items": {"type": "string"}}
      }
    },
    "deployments": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["what", "where", "deadline", "quantity"],
        "additionalProperties": false,
        "properties": {
          "what": {"type": "string"},
          "where": {"type": "array", "items": {"type": "string"}},
          "deadline": {"type": ["string", "null"]},
          "quantity": {"type": ["integer", "null"]}
        }
      }
    },
    "hiring": {
      "type": "object",
      "required": ["count", "focus_areas"],
      "additionalProperties": false,
      "properties": {
        "count": {"type": ["integer", "null"]},
        "focus_areas": {"type": "array", "items": {"type": "string"}}
      }
    },
    "financials": {
      "type": "object",
      "required": ["recurring_revenue_usd_millions",
                   "recurring_revenue_prior_year_usd_millions",
                   "gross_margin_percent", "cash_flow_positive_outlook"],
      "additionalProperties": false,
      "properties": {
        "recurring_revenue_usd_millions": {"type": ["number", "null"]},
        "recurring_revenue_prior_year_usd_millions": {"type": ["number", "null"]},
        "gross_margin_percent": {"type": ["number", "null"]},
        "cash_flow_positive_outlook": {"type": ["string", "null"]}
      }
    },
    "regulatory": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["authority", "jurisdiction", "topic", "named_companies"],
        "additionalProperties": false,
        "properties": {
          "authority": {"type": "string"},
          "jurisdiction": {"type": "string"},
          "topic": {"type": "string"},
          "named_companies": {"type": "boolean"}
        }
      }
    },
    "outage": {
      "type": "object",
      "required": ["date", "duration_minutes_reported",
                   "duration_minutes_initial_claim", "cause",
                   "affected_traffic_percent", "affected_seats",
                   "credits_disclosed"],
      "additionalProperties": false,
      "properties": {
        "date": {"type": ["string", "null"]},
        "duration_minutes_reported": {"type": ["number", "null"]},
        "duration_minutes_initial_claim": {"type": ["number", "null"]},
        "cause": {"type": ["string", "null"]},
        "affected_traffic_percent": {"type": ["number", "null"]},
        "affected_seats": {"type": ["string", "null"]},
        "credits_disclosed": {"type": "boolean"}
      }
    },
    "product_launches": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "notable_detail"],
        "additionalProperties": false,
        "properties": {
          "name": {"type": "string"},
          "notable_detail": {"type": ["string", "null"]}
        }
      }
    },
    "market": {
      "type": "object",
      "required": ["size_usd_billions", "growth_percent_yoy", "competitors"],
      "additionalProperties": false,
      "properties": {
        "size_usd_billions": {"type": ["number", "null"]},
        "growth_percent_yoy": {"type": ["number", "null"]},
        "competitors": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "city", "raise_usd_millions", "month"],
            "additionalProperties": false,
            "properties": {
              "name": {"type": "string"},
              "city": {"type": ["string", "null"]},
              "raise_usd_millions": {"type": ["number", "null"]},
              "month": {"type": ["string", "null"]}
            }
          }
        }
      }
    },
    "unconfirmed": {"type": "array", "items": {"type": "string"}}
  }
}
"""

_EXTRACTION_SOURCE = """\
Hai Duong, Vietnam - Thursday, 09:12 ICT. Meridian Compute, a Hanoi-based
provider of managed inference, said it closed a Series B of $86 million led by
Northbank Growth with participation from existing investor Sao Do Ventures and a
new entrant, Kisen Partners. The round values Meridian at $610 million pre-money,
according to two people familiar with the matter, and is conditioned on the
completion of an audit of its data processing practices.

Chief executive Le Thu Ha said the money would be used to deploy 240 additional
GPU nodes across two facilities in Hai Phong and Singapore before the end of the
third quarter, and to hire 60 engineers, mostly in reliability and security.
"We are not buying model capability, we are buying headroom," Ha said at a
briefing. Meridian reported recurring revenue of $19.4 million in the twelve
months to June, up from $7.8 million the prior year, and a gross margin of 41
percent, weighed down by cloud resale.

Competitive context: regional rivals have been raising too. Jakarta's Sangir AI
announced $52 million in February, and Bangkok-based Lom Data raised $140 million
in a debt facility in April. Analysts at Veristat put the regional managed
inference market at $1.1 billion in annualised revenue, growing about 34 percent
year over year.

Separately, regulators in two of Meridian's markets opened inquiries. The
Vietnamese competition authority confirmed it is reviewing managed inference
procurement contracts, without naming companies, after complaints that bundled
pricing foreclosed smaller local providers. Singapore's IMDA said it requested
documentation from several providers about incident notification practices
following an outage on 4 June that lasted roughly 47 minutes and affected
"several thousand" enterprise seats, according to a status page post that was
later corrected from an initial figure of 12 minutes.

Meridian said the outage was caused by a misconfigured rolling update to a router
tier and affected 0.6 percent of total traffic. The company disclosed service
credits but not their value. An unnamed customer said the credit covered "less
than a day of spend" while their own recovery took eleven hours.

On the product side, Meridian launched three items this week: an OpenAI compatible
gateway with per-key budgets; a batch endpoint priced at 55 percent of interactive
rates; and a benchmarking service that publishes per-category throughput numbers,
initially covering coding, summarisation and translation. Product lead Pham Minh
said the benchmarking service will add structured extraction and reasoning "once
the prompts stop being our problem".

Two financial details remain unconfirmed: whether the Series B includes $12
million of second-close capital already committed, and whether the Singapore
facility lease includes a power escalator clause tied to industrial tariff
changes. Meridian declined to comment on both. Ha said the company expects to be
cash flow positive "in the second half of next year" and repeated that the
company has no plans to raise debt.
"""

_EXTRACTION_INSTRUCTION = (
    "Extract structured data from the article below using the JSON Schema"
    " provided.\n\nRules:\n"
    "- Output a single JSON object and nothing else. No Markdown fence, no"
    " prose, no trailing commas, no comments, no keys outside the schema.\n"
    "- Use null for any required field whose value is genuinely absent from the"
    " article. Never guess, never infer a number that is not stated, and never"
    " derive a figure arithmetically unless the article states the"
    " derivation.\n"
    "- Convert money to the numeric unit named in the key (millions for"
    " *_usd_millions, billions for *_usd_billions, percent as a plain"
    " number).\n"
    "- participating_investors lists every named investor other than the"
    " lead.\n"
    "- competitors lists only explicitly named rival companies, with their"
    " raise amount and month when stated.\n"
    "- regulatory contains one entry per distinct authority action"
    " described.\n"
    "- unconfirmed lists each fact the article itself flags as unconfirmed,"
    " unnamed, declined-to-comment, or corrected after being wrong.\n"
    "- affected_seats must reproduce the article's own hedge, verbatim, as a"
    " string.\n"
    "- Do not include the reporter's speculation, and do not merge the"
    " corrected outage duration with the initial claim: they belong in"
    " separate fields.\n\nJSON Schema:\n```json\n" + _EXTRACTION_SCHEMA +
    "```\n\nArticle:\n" + _EXTRACTION_SOURCE
)

# explain / deep-explain: ~700 input tokens
_EXPLAIN_DEEP = """\
You are writing an internal architecture note for a team that will operate a
private text-only LLM inference platform: two or four GPU boxes, one or two
models at a time, an OpenAI-compatible endpoint, and a small internal user base
of engineers.

Audience: competent software engineers with no prior LLM serving experience.
Length: 900 to 1300 words. Structure: exactly the seven numbered sections below,
in order, with the given headings, plus a closing "Recommended defaults" table.

1. Prefill and decode as separate engineering problems
   Explain why the two phases have different bottlenecks, what each one is
   sensitive to, and why a single blended "tokens per second" number hides the
   thing you care about. Give one concrete example where optimising the blended
   number makes user experience worse.

2. How concurrency actually changes both numbers
   Explain batching, why decode throughput per sequence is roughly flat until
   the memory system saturates, and why prefill throughput per sequence falls
   roughly in proportion to concurrency. Explain how queueing turns into
   time-to-first-token, and why measuring at concurrency 1 tells you almost
   nothing about peak-hour behaviour.

3. Key value cache and context length as a budget
   Explain what consumes memory during inference, how context length and
   concurrent sequences multiply, and the failure mode where the server admits
   more sequences than it can hold. Cover prefix caching and quantised KV, with
   their quality risks.

4. Quantisation choices for this fleet
   Compare weight-only four bit against FP8 style formats for the two phases,
   and specifically discuss which task categories break first: long chain
   reasoning, exact instruction following, and structured JSON output. Explain
   how to detect silent quality regressions with a prompt suite rather than a
   single average score.

5. A measurement plan
   Describe how to build a prompt suite that reflects real traffic, which task
   categories to include, how long prompts should be for prefill measurement to
   be meaningful, whether to run sequentially or concurrently, and how to read
   the mean of per-request ratios versus aggregate throughput. State what to do
   when those two disagree.

6. Routing and why it breaks your numbers
   The team intends to run one small fast model for translation and drafting and
   one large model for coding and reasoning, behind a single endpoint. Explain
   what this does to measured throughput numbers, why a benchmark that silently
   measures whichever model an alias resolves to is dangerous, and how to pin the
   model explicitly so a routing change cannot be mistaken for a regression.

7. Operational failure modes
   Cover, concretely: a model that streams everything in one burst, reasoning or
   thinking tokens being counted as output, missing usage data from the server,
   prompt length exceeding context, cold-server effects on the first
   measurement, and throughput collapse at a specific concurrency cliff.

Recommended defaults: a Markdown table with columns Setting | Default | Why,
covering max prompt length, max output tokens, warmup requests, sequential
versus concurrent default, prompt suite composition, and which throughput number
to use for capacity planning versus user experience. State one default you expect
to disagree with internally and why.

Also required, inside section 5: define the three metrics your team will publish
on a dashboard (name, unit, how it is computed, what a bad value looks like), and
state the alerting rule you would attach to each. Explain what to do when the
mean and the aggregate of the same measurement diverge, and give one realistic
cause of divergence that is not a bug.

Style: direct, no marketing language, no bullet spam (no more than five bullets
total across the whole note), prefer short declarative sentences, and define
every abbreviation on first use. Do not invent hardware specifications, vendor
benchmarks, or prices that are not given in this brief; when a number is needed
but unknown, say what measurement would produce it.
"""

PROMPTS = [
    {
        "category": "coding",
        "name": "implement-module",
        "text": _CODING_IMPLEMENT_MODULE,
    },
    {
        "category": "code-review",
        "name": "review-refactor",
        "text": _CODE_REVIEW_REFACTOR,
    },
    {
        "category": "translation",
        "name": "en-vi-article",
        "text": _TRANSLATION_EN_VI,
    },
    {
        "category": "translation",
        "name": "vi-en-article",
        "text": _TRANSLATION_VI_EN,
    },
    {
        "category": "summarization",
        "name": "meeting-minutes",
        "text": _SUMMARIZE_TRANSCRIPT,
    },
    {
        "category": "creative-writing",
        "name": "scene-dossier",
        "text": _CREATIVE_WRITING_DOSSIER,
    },
    {
        "category": "reasoning",
        "name": "multi-step-analysis",
        "text": _REASONING_CAPACITY,
    },
    {
        "category": "structured-extraction",
        "name": "json-schema",
        "text": _EXTRACTION_INSTRUCTION,
    },
    {
        "category": "explain",
        "name": "deep-explain",
        "text": _EXPLAIN_DEEP,
    },
]


@dataclass
class RequestResult:
  """Measurement outcome for a single prompt."""

  category: str
  name: str
  prompt_tokens: int
  usage_ok: bool
  prefill_s: float
  decode_s: float
  ttft_ms: float
  input_tps: float
  gen_tokens: int
  output_tps: float
  failed: bool = False
  error: str = ""
  burst_capped: bool = False


def cache_bust(prompt_spec: dict) -> dict:
  """Returns the prompt with a unique nonce prepended.

  Servers with prefix caching (llama.cpp, vLLM) will serve a repeated prompt
  from cache, so a second pass reports a near-instant prefill and an input TPS
  number that is pure fiction. The nonce makes every prompt a cold prefix. It
  costs roughly ten prompt tokens, which is noise at these prompt lengths.
  """
  return {
      **prompt_spec,
      "text": (
          f"Benchmark run id: {secrets.token_hex(4)}. This line is measurement"
          " noise; ignore it.\n\n" + prompt_spec["text"]
      ),
  }


def estimate_tokens(text: str) -> int:
  """Rough token estimate, used only for warnings and --list-prompts."""
  return max(1, int(len(text.split()) * WORDS_TO_TOKENS))


def chunked(items: list, size: int) -> list:
  """Split items into consecutive chunks of at most `size` elements."""
  if size < 1:
    raise ValueError("chunk size must be >= 1")
  return [items[i:i + size] for i in range(0, len(items), size)]


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
    client: AsyncOpenAI,
    prompt_spec: dict,
    model: str,
    max_tokens: int = 256,
) -> RequestResult:
  """Executes a single async streaming request for one prompt spec.

  Metrics:
    TTFT       : time from request start until first content token arrives.
    Input TPS  : prompt_tokens / prefill_time. prefill_time == TTFT (in
                 streaming, prefill happens before the first token emits).
    Output TPS : gen_tokens / decode_time. decode_time is the wall-clock span
                 from the first content token to the last content token, so
                 it measures actual generation throughput.
  """
  category = prompt_spec["category"]
  name = prompt_spec["name"]
  prompt = prompt_spec["text"]

  t_start = time.perf_counter()
  t_first = None      # arrival time of first content token
  t_last = None       # arrival time of last content token
  gen_tokens = 0      # content tokens counted from stream
  prompt_tokens = 0
  usage_ok = False

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
          usage_ok = True
        if chunk.usage.completion_tokens:
          gen_tokens = chunk.usage.completion_tokens

    t_end = time.perf_counter()

    # Server usage wins. Fall back to a word estimate when the backend does
    # not return usage (llama.cpp builds vary by version) and flag the row.
    if not usage_ok:
      prompt_tokens = estimate_tokens(prompt)

    # Prefer stream-derived timings; fall back to the closing timestamp.
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
    burst_capped = (t_last - t_first) <= 1e-9 and gen_tokens > 1
    output_tps = gen_tokens / decode_s if gen_tokens > 0 else 0.0

    return RequestResult(
        category=category,
        name=name,
        prompt_tokens=prompt_tokens,
        usage_ok=usage_ok,
        prefill_s=prefill_s,
        decode_s=decode_s,
        ttft_ms=ttft_ms,
        input_tps=input_tps,
        gen_tokens=gen_tokens,
        output_tps=output_tps,
        burst_capped=burst_capped,
    )

  except Exception as e:
    return RequestResult(
        category=category,
        name=name,
        prompt_tokens=estimate_tokens(prompt),
        usage_ok=False,
        prefill_s=0.0,
        decode_s=0.0,
        ttft_ms=0.0,
        input_tps=0.0,
        gen_tokens=0,
        output_tps=0.0,
        failed=True,
        error=str(e),
    )


def print_report(results: list, verbose: bool = False) -> int:
  """Prints per-prompt (optional), per-category, mean and aggregate numbers."""
  valid = [r for r in results if not r.failed]
  failed = [r for r in results if r.failed]

  if not valid:
    print("❌ All benchmark requests failed!")
    return 1

  if verbose:
    print("=" * 80)
    print("📄 PER-PROMPT RESULTS")
    print("=" * 80)
    print(
        f"{'category':<23}{'prompt':<17}{'in tok':>10}{'TTFT ms':>10}"
        f"{'in tok/s':>12}{'out tok':>9}{'out tok/s':>14}"
    )
    for r in valid:
      # A trailing ~ marks a row where the server returned no usage data and
      # prompt_tokens is a word-count estimate.
      tok_cell = f"{r.prompt_tokens}{'~' if not r.usage_ok else ''}"
      flag = " ⚡burst" if r.burst_capped else ""
      print(
          f"{r.category:<23}{r.name:<17}{tok_cell:>10}"
          f"{r.ttft_ms:>10.2f}{r.input_tps:>12.2f}"
          f"{r.gen_tokens:>9}{r.output_tps:>14.2f}{flag}"
      )

  # Per-category rollup, one row per category in first-seen order.
  categories: list = []
  by_category: dict = {}
  for r in valid:
    if r.category not in by_category:
      by_category[r.category] = []
      categories.append(r.category)
    by_category[r.category].append(r)

  print("=" * 80)
  print("📈 BENCHMARK RESULTS SUMMARY")
  print("=" * 80)
  print(
      f"{'category':<23}{'#prompts':>9}{'avg in tok':>12}{'avg TTFT ms':>13}"
      f"{'avg in tok/s':>14}{'avg out tok/s':>16}"
  )
  for category in categories:
    rows = by_category[category]
    count = len(rows)
    print(
        f"{category:<23}{count:>9}"
        f"{sum(r.prompt_tokens for r in rows) / count:>12.1f}"
        f"{sum(r.ttft_ms for r in rows) / count:>13.2f}"
        f"{sum(r.input_tps for r in rows) / count:>14.2f}"
        f"{sum(r.output_tps for r in rows) / count:>16.2f}"
    )

  count = len(valid)
  avg_ttft = sum(r.ttft_ms for r in valid) / count
  avg_in_tps = sum(r.input_tps for r in valid) / count
  avg_out_tps = sum(r.output_tps for r in valid) / count

  # Aggregate = total tokens over total time. This is the honest throughput
  # number when prompt lengths differ. The overall mean above is the mean of
  # per-request ratios, kept so runs stay comparable with earlier baselines.
  total_prefill_s = sum(r.prefill_s for r in valid)
  total_decode_s = sum(r.decode_s for r in valid)
  agg_in_tps = (
      sum(r.prompt_tokens for r in valid) / total_prefill_s
      if total_prefill_s > 0
      else 0.0
  )
  agg_out_tps = (
      sum(r.gen_tokens for r in valid) / total_decode_s
      if total_decode_s > 0
      else 0.0
  )

  print("-" * 80)
  print(f"✅ Requests OK     : {count} / {len(results)}")
  print(f"⚡ Avg TTFT        : {avg_ttft:.2f} ms")
  print(f"📥 Avg Input TPS   : {avg_in_tps:.2f} tok/s")
  print(f"📤 Avg Output TPS  : {avg_out_tps:.2f} tok/s")
  print("-" * 80)
  print(f"🧮 Aggregate Input : {agg_in_tps:.2f} tok/s "
        f"(sum tok / {total_prefill_s:.2f} s prefill)")
  print(f"🧮 Aggregate Output: {agg_out_tps:.2f} tok/s "
        f"(sum tok / {total_decode_s:.2f} s decode)")

  missing_usage = sum(1 for r in valid if not r.usage_ok)
  if missing_usage:
    print(
        f"⚠️ {missing_usage} prompt(s) had no server usage counts "
        f"(marked ~): input TPS approximate for those rows."
    )
  burst_rows = [r for r in valid if r.burst_capped]
  if burst_rows:
    print(
        f"⚠️ {len(burst_rows)} prompt(s) streamed in a single burst: "
        f"output TPS reflects server batching, not decode speed."
    )
  for r in failed:
    print(f"❌ Failed: {r.category}/{r.name} ({r.error})")

  print("=" * 80)
  return 0


def print_prompt_catalog() -> None:
  """Lists the prompt suite without contacting the server."""
  print(f"{'category':<23}{'prompt':<17}{'approx in tok':>14}{'words':>8}")
  total_words = 0
  for spec in PROMPTS:
    words = len(spec["text"].split())
    total_words += words
    print(
        f"{spec['category']:<23}{spec['name']:<17}"
        f"{estimate_tokens(spec['text']):>14}{words:>8}"
    )
  print("-" * 60)
  print(
      f"{len(PROMPTS)} prompts, {len({p['category'] for p in PROMPTS})} "
      f"categories, {total_words} words total "
      f"(x{WORDS_TO_TOKENS} ≈ {int(total_words * WORDS_TO_TOKENS)} tokens)"
  )


def warn_context_length(max_tokens: int) -> None:
  """Warns when prompt + generation budget may exceed the smallest context."""
  offenders = [
      spec
      for spec in PROMPTS
      if estimate_tokens(spec["text"]) + max_tokens > CONSERVATIVE_MAX_MODEL_LEN
  ]
  if offenders:
    names = ", ".join(f"{s['category']}/{s['name']}" for s in offenders)
    print(
        f"⚠️ With --max-tokens {max_tokens}, these prompts may exceed "
        f"{CONSERVATIVE_MAX_MODEL_LEN} tokens (conservative floor used by "
        f"the smallest launcher in this repo): {names}"
    )


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
      help="Prompts fired at once. The full suite is run in consecutive "
           "chunks of this size until every prompt has been measured once "
           "(default: 1, i.e. strictly sequential)",
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=256,
      help="Max generation tokens per prompt",
  )
  parser.add_argument(
      "--warmup",
      type=int,
      default=1,
      help="Throwaway requests before measurement, excluded from all stats "
           "(default: 1, use 0 to disable)",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Also print one row per prompt above the per-category summary",
  )
  parser.add_argument(
      "--cache-bust",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Prepend a unique nonce to every prompt so servers with prefix "
           "caching measure a real cold prefill on every run. Use "
           "--no-cache-bust to measure warm prefix caching instead "
           "(default: enabled)",
  )
  parser.add_argument(
      "--list-prompts",
      action="store_true",
      help="List the prompt suite (category, name, approx input tokens) and "
           "exit without contacting the server",
  )
  args = parser.parse_args()

  if args.list_prompts:
    print_prompt_catalog()
    return 0

  if args.concurrency < 1:
    print("❌ --concurrency must be >= 1")
    return 1

  client = AsyncOpenAI(base_url=args.url, api_key=args.key)

  # Auto-resolve model if not explicitly provided
  model_name = args.model
  if not model_name:
    model_name = await fetch_first_model(client)

  chunks = chunked(PROMPTS, args.concurrency)
  print(f"\n🚀 Starting Benchmark against: {args.url}")
  print(
      f"📊 Model: {model_name} | Prompts: {len(PROMPTS)} in "
      f"{len(chunks)} chunk(s) of {args.concurrency} | Max Tokens:"
      f" {args.max_tokens} | Warmup: {args.warmup} | Cache bust:"
      f" {'on' if args.cache_bust else 'off'}\n"
  )
  if not args.cache_bust:
    print(
        "ℹ️ --no-cache-bust: repeating this run may report inflated input "
        "TPS because the server serves cached prefixes."
    )
  warn_context_length(args.max_tokens)

  for i in range(args.warmup):
    await benchmark_single_request(
        client,
        {"category": "warmup", "name": f"warmup-{i + 1}", "text": WARMUP_PROMPT},
        model_name,
        max_tokens=8,
    )

  results: list = []
  interrupted = False
  try:
    for index, chunk in enumerate(chunks, start=1):
      print(
          f"▶ Chunk {index}/{len(chunks)}: "
          + ", ".join(spec["name"] for spec in chunk)
      )
      specs = ([cache_bust(spec) for spec in chunk] if args.cache_bust
               else chunk)
      batch = await asyncio.gather(
          *[
              benchmark_single_request(client, spec, model_name, args.max_tokens)
              for spec in specs
          ]
      )
      results.extend(batch)
  except (KeyboardInterrupt, asyncio.CancelledError):
    interrupted = True
    print("\n⚠️ Interrupted: reporting chunks completed so far")

  if not results:
    print(
        "❌ Interrupted before any chunk completed"
        if interrupted
        else "❌ All benchmark requests failed!"
    )
    return 1

  return print_report(results, verbose=args.verbose)


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
