"""
性能对比仪表盘生成器

负责解析Structured Context和PlainText Context两个版本的运行报告，
生成HTML格式的性能对比仪表盘。
主要功能：
- 查找最新的报告文件
- 解析报告中的token使用量和耗时指标
- 计算优化百分比
- 生成可视化HTML仪表盘
"""

import re
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
ACP_DIR = ROOT / "ACP_hackathon"
BASELINE_DIR = ROOT / "baseline"
OUTPUT_HTML = ROOT / "comparison_dashboard.html"

REPORT_PATTERN = re.compile(r"^\d{8}_\d{6}\.txt$")
TOKEN_PATTERN = re.compile(r"^total_tokens:\s*(\d+)\s*$", re.IGNORECASE)
ELAPSED_PATTERN = re.compile(r"^elapsed_seconds:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)


def find_latest_report(version_dir: Path) -> Optional[Path]:
    reports = [p for p in version_dir.glob("*.txt") if REPORT_PATTERN.match(p.name)]
    if not reports:
        return None
    return max(reports, key=lambda p: p.stem)


def parse_metrics(report_path: Path) -> Tuple[int, float]:
    total_tokens = None
    elapsed_seconds = None

    for line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        token_match = TOKEN_PATTERN.match(line.strip())
        if token_match:
            total_tokens = int(token_match.group(1))
            continue

        elapsed_match = ELAPSED_PATTERN.match(line.strip())
        if elapsed_match:
            elapsed_seconds = float(elapsed_match.group(1))

    if total_tokens is None or elapsed_seconds is None:
        raise ValueError(f"Missing metrics in report: {report_path}")

    return total_tokens, elapsed_seconds


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def build_html(
    generated_at: str,
    acp_report_rel: str,
    baseline_report_rel: str,
    acp_tokens: int,
    acp_elapsed: float,
    baseline_tokens: int,
    baseline_elapsed: float,
    token_reduction_pct: float,
    time_reduction_pct: float,
) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Agent Context Interaction Optimization Comparison</title>
  <style>
    :root {{
      --bg1: #f6f7fb;
      --bg2: #e6ecff;
      --card: #ffffff;
      --text: #1b1f2a;
      --muted: #596279;
      --ok: #18794e;
      --accent: #2346a0;
      --border: #d9e0f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background: linear-gradient(160deg, var(--bg1), var(--bg2));
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 24px;
    }}
    .wrap {{ width: min(980px, 100%); }}
    .hero {{ margin-bottom: 16px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 30px; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 10px 28px rgba(25, 40, 75, 0.08);
    }}
    .title {{ margin: 0 0 12px 0; font-size: 18px; color: var(--accent); }}
    .k {{ color: var(--muted); font-size: 14px; }}
    .v {{ font-size: 28px; font-weight: 700; margin: 4px 0 10px 0; }}
    .ok {{ color: var(--ok); }}
    .meta {{ margin-top: 14px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>Agent Context Interaction Optimization: Performance Dashboard</h1>
      <p class=\"sub\">Generated at {generated_at}</p>
    </div>

    <div class=\"card\">
      <h2 class=\"title\">Comparison Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Version</th>
            <th>Total Tokens</th>
            <th>Task Time (seconds)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Structured Context</td>
            <td>{acp_tokens:,}</td>
            <td>{fmt_num(acp_elapsed)}</td>
          </tr>
          <tr>
            <td>PlainText Context</td>
            <td>{baseline_tokens:,}</td>
            <td>{fmt_num(baseline_elapsed)}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class=\"grid\">
      <div class=\"card\">
        <p class=\"k\">Structured Context Token Reduction vs PlainText Context</p>
        <p class=\"v ok\">{fmt_num(token_reduction_pct)}%</p>
      </div>
      <div class=\"card\">
        <p class=\"k\">Structured Context Time Reduction vs PlainText Context</p>
        <p class=\"v ok\">{fmt_num(time_reduction_pct)}%</p>
      </div>
    </div>

    <div class=\"meta\">
      <div>Structured Context source: {acp_report_rel}</div>
      <div>PlainText Context source: {baseline_report_rel}</div>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    acp_report = find_latest_report(ACP_DIR)
    baseline_report = find_latest_report(BASELINE_DIR)

    if acp_report is None:
        raise FileNotFoundError(f"No timestamp report found in {ACP_DIR}")
    if baseline_report is None:
        raise FileNotFoundError(f"No timestamp report found in {BASELINE_DIR}")

    acp_tokens, acp_elapsed = parse_metrics(acp_report)
    baseline_tokens, baseline_elapsed = parse_metrics(baseline_report)

    token_reduction_pct = ((baseline_tokens - acp_tokens) / baseline_tokens * 100.0) if baseline_tokens else 0.0
    time_reduction_pct = ((baseline_elapsed - acp_elapsed) / baseline_elapsed * 100.0) if baseline_elapsed else 0.0

    html = build_html(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        acp_report_rel=str(acp_report.relative_to(ROOT)),
        baseline_report_rel=str(baseline_report.relative_to(ROOT)),
        acp_tokens=acp_tokens,
        acp_elapsed=round(acp_elapsed, 2),
        baseline_tokens=baseline_tokens,
        baseline_elapsed=round(baseline_elapsed, 2),
        token_reduction_pct=round(token_reduction_pct, 2),
        time_reduction_pct=round(time_reduction_pct, 2),
    )

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print("Saved dashboard:", OUTPUT_HTML)
    opened = webbrowser.open(OUTPUT_HTML.resolve().as_uri())
    if not opened:
        print(f"Could not auto-open browser. Please open manually: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()