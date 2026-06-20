"""Fermat Plugin for AstrBot — Scientific computing & plotting.

LLM tools:
  - fermat_plot_function : Plot functions (just pass expressions)  → image
  - fermat_draw          : Free-form Matplotlib drawing            → image
  - fermat_compute       : NumPy / SciPy / SymPy code execution   → text
  - fermat_analyze       : Pandas data analysis                   → text
  - fermat_quick_compute : Evaluate a math expression             → text

All tools run with libraries pre-imported
(np, sp, sympy, pd, plt, sin, cos, sqrt, pi, e, ...).
"""

import uuid
from pathlib import Path
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .fmcp.executor import execute
from .fmcp.image import Image


# ── helpers ──────────────────────────────────────────────────────────

def _err(msg: str) -> str:
    """Extract the last meaningful line from an error traceback."""
    lines = [l for l in msg.strip().split("\n") if l.strip()]
    return lines[-1] if lines else msg.strip()[:200]


def _usage() -> str:
    return (
        "Fermat 科学计算 v1.3.0\n\n"
        "自然语言直接用：\n"
        "  画个 y=x^2\n"
        "  画 sin(x) 和 cos(x) 在 -5 到 5\n"
        "  用 matplotlib 画个猫咪\n"
        "  求 [[1,2],[3,4]] 的逆矩阵\n"
        "  分析数据：...\n\n"
        "命令行：\n"
        "  /fermat compute <python代码>\n"
        "  /fermat draw <matplotlib代码>\n"
    )


# ── plugin class ─────────────────────────────────────────────────────

@register(
    "astrbot_plugin_fermat",
    "OpenAI Codex + abhiphile",
    "科学计算与绘图：NumPy/SciPy/SymPy/Pandas/Matplotlib，一行表达式即可画图。",
    "1.3.0",
)
class FermatPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.output_dir = Path(__file__).resolve().parent / "generated"
        self.output_dir.mkdir(exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────

    MAX_CONTEXT = 300

    def _summarize(self, text: str) -> str:
        t = text.strip()
        return t if len(t) <= self.MAX_CONTEXT else t[:self.MAX_CONTEXT] + "\n...(已截断)"

    def _img_summary(self, fallback: str, titles: list[str] | None = None) -> str:
        if titles:
            real = [t for t in titles if t]
            if real:
                return f"[已生成图像: {', '.join(real)}]"
        return f"[已生成图像: {fallback}]"

    def _save(self, image: Image) -> str:
        fmt = getattr(image, "format", "png") or "png"
        path = self.output_dir / f"fermat_{uuid.uuid4().hex}.{fmt}"
        path.write_bytes(image.data)
        return str(path)

    def _emit_image(self, event: AstrMessageEvent, fallback: str,
                    images: list, titles: list[str] | None = None):
        """Yield a chain result with summary text + images."""
        items = [Comp.Plain(self._img_summary(fallback, titles))]
        for _, img in images:
            items.append(Comp.Image.fromFileSystem(self._save(img)))
        return event.chain_result(items)

    # ── /fermat command ──────────────────────────────────────────

    @filter.command("fermat")
    async def fermat(self, event: AstrMessageEvent):
        msg = event.message_str.strip()
        parts = msg.split(maxsplit=2)

        if len(parts) == 1 or parts[1].lower() in ("help", "帮助"):
            yield event.plain_result(_usage())
            return

        sub = parts[1].lower()
        if len(parts) < 3:
            yield event.plain_result("缺少代码参数。\n\n" + _usage())
            return

        code = parts[2]
        out = execute(code, self.output_dir)
        if out["error"]:
            yield event.plain_result(f"执行出错：\n{out['error']}")
        elif out["images"]:
            yield self._emit_image(event, sub, out["images"], out.get("titles"))
        else:
            text = out["text"] or "(无输出)"
            yield event.plain_result(self._summarize(text))

    # ═════════════════════════════════════════════════════════════
    # LLM Tools
    # ═════════════════════════════════════════════════════════════

    # ── drawing tools ────────────────────────────────────────────

    @filter.llm_tool(name="fermat_plot_function")
    async def fermat_plot_function(
        self, event: AstrMessageEvent,
        expressions: str,
        x_min: float = -10.0,
        x_max: float = 10.0,
        y_min: float | None = None,
        y_max: float | None = None,
        title: str = "",
        grid: bool = True,
    ):
        """Plot mathematical functions. Use this FIRST for any graphing request.

        The SIMPLEST tool — just pass expressions, no code needed.
        Examples: "x**2", "sin(x), cos(x)", "x**2, x**3, sqrt(x)"

        Args:
            expressions(string): Function expression(s), comma-separated. e.g. "x**2" or "x**2, sin(x)".
            x_min(number): Left x bound (default -10).
            x_max(number): Right x bound (default 10).
            y_min(number): Lower y bound, auto if omitted.
            y_max(number): Upper y bound, auto if omitted.
            title(string): Plot title, auto if omitted.
            grid(boolean): Show grid (default true).
        """
        funcs = [f.strip() for f in expressions.split(",") if f.strip()]
        if not funcs:
            yield event.plain_result("请提供至少一个函数表达式，如 x**2 或 sin(x)")
            return

        labels = ", ".join(funcs)
        plot_title = repr(title) if title else repr(labels)
        ylim_line = ""
        if y_min is not None and y_max is not None:
            ylim_line = f"plt.ylim({y_min}, {y_max})"
        elif y_min is not None:
            ylim_line = f"plt.ylim(bottom={y_min})"
        elif y_max is not None:
            ylim_line = f"plt.ylim(top={y_max})"

        code = (
            f"x = linspace({x_min}, {x_max}, 500)\n"
            + "\n".join(f"plt.plot(x, {f}, linewidth=2, label={repr(f)})" for f in funcs)
            + f"\nif {len(funcs) > 1}: plt.legend()\n"
            + f"plt.title({plot_title})\n"
            + "plt.xlabel('x'); plt.ylabel('y')\n"
            + (f"{ylim_line}\n" if ylim_line else "")
            + ("plt.grid(True, alpha=0.3)\n" if grid else "")
        )

        out = execute(code, self.output_dir)
        if out["error"]:
            yield event.plain_result(f"绘图失败：{_err(out['error'])}")
            return
        if out["images"]:
            rng = f"x[{x_min},{x_max}]"
            yield self._emit_image(event, f"{labels} 在 {rng}", out["images"], out.get("titles"))
        else:
            yield event.plain_result("未生成图像")

    @filter.llm_tool(name="fermat_draw")
    async def fermat_draw(self, event: AstrMessageEvent, code: str):
        """Free-form Matplotlib drawing. Use for custom/creative plots.

        Pre-imported: plt, np, sin, cos, sqrt, pi, e, linspace, arange, etc.
        DO NOT call plt.show() or plt.savefig() — auto-captured.
        ALWAYS set plt.title("description") so context stays meaningful.

        Args:
            code(string): Matplotlib Python code.
        """
        out = execute(code, self.output_dir)
        if out["error"]:
            yield event.plain_result(f"绘图失败：{_err(out['error'])}")
            return
        if out["images"]:
            yield self._emit_image(event, "自定义图形", out["images"], out.get("titles"))
        else:
            yield event.plain_result("未生成图像，请检查代码是否创建了图形。")

    # ── compute tools ────────────────────────────────────────────

    @filter.llm_tool(name="fermat_compute")
    async def fermat_compute(self, event: AstrMessageEvent, code: str):
        """Execute Python code for math / science / engineering computation.

        Pre-imported: np (NumPy), sp (SciPy), sympy (SymPy),
        sin, cos, tan, sqrt, log, exp, pi, e, linspace, arange, etc.
        Use print() to output. Result is always returned as text.

        Args:
            code(string): Python code to execute.
        """
        out = execute(code, self.output_dir)
        if out["error"]:
            return f"计算失败：{_err(out['error'])}"
        return self._summarize(out["text"] or "(无输出)")

    @filter.llm_tool(name="fermat_analyze")
    async def fermat_analyze(self, event: AstrMessageEvent, code: str):
        """Execute Python code for data analysis with Pandas.

        Pre-imported: pd (Pandas), np (NumPy), sp (SciPy).
        Use print() to output. Result is always returned as text.

        Args:
            code(string): Pandas data analysis code.
        """
        out = execute(code, self.output_dir)
        if out["error"]:
            return f"数据分析失败：{_err(out['error'])}"
        return self._summarize(out["text"] or "(无输出)")

    @filter.llm_tool(name="fermat_quick_compute")
    async def fermat_quick_compute(self, event: AstrMessageEvent, expression: str):
        """Evaluate a single math expression and return the number.

        For quick calculations. Pre-imported: np, sin, cos, sqrt, pi, e, etc.

        Args:
            expression(string): Math expression, e.g. "sin(pi/2)" or "sqrt(16)+3*5".
        """
        out = execute(f"result = {expression}\nprint(result)", self.output_dir)
        if out["error"]:
            return f"计算失败：{_err(out['error'])}"
        return self._summarize(out["text"].strip() if out["text"] else "(无输出)")

    # ── lifecycle ────────────────────────────────────────────────

    async def terminate(self):
        pass
