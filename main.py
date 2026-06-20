"""Fermat Plugin for AstrBot — Scientific computing & plotting via code execution.

LLM tools:
  - fermat_plot_function : Plot functions by expression string → image
  - fermat_solve_math    : Symbolic math (diff/integrate/solve) → text
  - fermat_quick_compute : Evaluate an expression numerically   → text
  - fermat_compute       : NumPy / SciPy / SymPy code          → text
  - fermat_draw          : Matplotlib drawing code             → image
  - fermat_analyze       : Pandas data analysis code           → text

All compute/draw/analyze tools run with libraries pre-imported
(np, sp, sympy, pd, plt) so the LLM only writes the logic.
"""

import uuid
from pathlib import Path
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .fmcp.executor import execute
from .fmcp.image import Image

# ---------------------------------------------------------------------------
# old-style command tools (kept for backward compatibility & /fermat command)
# ---------------------------------------------------------------------------
from .fmcp.mpl_mcp.core.bar_chart import plot_barchart
from .fmcp.mpl_mcp.core.eqn_chart import eqn_chart
from .fmcp.mpl_mcp.core.plot_chart import plot_chart
from .fmcp.mpl_mcp.core.scatter_chart import plot_scatter
from .fmcp.mpl_mcp.core.stack_chart import plot_stack
from .fmcp.mpl_mcp.core.stem_chart import plot_stem
from .fmcp.numpy_mcp.core.matlib import matlib_operation
from .fmcp.numpy_mcp.core.numerical_operation import numerical_operation
from .fmcp.sympy_mcp.core.algebra import algebra_operation
from .fmcp.sympy_mcp.core.calculus import calculus_operation
from .fmcp.sympy_mcp.core.equations import equation_operation
from .fmcp.sympy_mcp.core.matrices import matrix_operation

import json

TEXT_TOOLS: dict[tuple[str, str], Any] = {
    ("numpy", "numerical"): numerical_operation,
    ("numpy", "matlib"): matlib_operation,
    ("sympy", "algebra"): algebra_operation,
    ("sympy", "calculus"): calculus_operation,
    ("sympy", "equation"): equation_operation,
    ("sympy", "matrix"): matrix_operation,
}

IMAGE_TOOLS: dict[str, Any] = {
    "bar": plot_barchart,
    "scatter": plot_scatter,
    "chart": plot_chart,
    "stem": plot_stem,
    "stack": plot_stack,
    "equation": eqn_chart,
}


def _parse_payload(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON 参数必须是对象")
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _usage() -> str:
    return (
        "Fermat 科学计算插件 v1.2.0\n\n"
        "=== 自然语言（推荐）===\n"
        "  画个 sin(x) 和 cos(x)\n"
        "  求 x^3 的二阶导数\n"
        "  用 matplotlib 画个猫咪\n"
        "  用 NumPy 求矩阵的逆\n"
        "  用 pandas 分析数据\n\n"
        "=== 命令行 ===\n"
        "/fermat compute <python代码>\n"
        "/fermat draw <matplotlib代码>\n"
        "/fermat analyze <pandas代码>\n"
        "/fermat sympy algebra {\"operation\":\"simplify\",\"expr\":\"x+x\"}\n\n"
        "集成库：numpy, scipy, sympy, pandas, matplotlib"
    )


# ======================================================================
@register(
    "astrbot_plugin_fermat",
    "OpenAI Codex + abhiphile",
    "科学计算与绘图插件：NumPy / SciPy / SymPy / Pandas / Matplotlib。支持函数绘图、符号计算、数值运算、数据分析。",
    "1.2.0",
)
class FermatPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.output_dir = Path(__file__).resolve().parent / "generated"
        self.output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_image(self, image: Image) -> str:
        fmt = getattr(image, "format", "png") or "png"
        path = self.output_dir / f"fermat_{uuid.uuid4().hex}.{fmt}"
        path.write_bytes(image.data)
        return str(path)

    def _image_result(self, event: AstrMessageEvent, path: str):
        return event.chain_result([Comp.Image.fromFileSystem(path)])

    # ------------------------------------------------------------------
    # /fermat command
    # ------------------------------------------------------------------
    @filter.command("fermat")
    async def fermat(self, event: AstrMessageEvent):
        message = event.message_str.strip()
        parts = message.split(maxsplit=3)

        if len(parts) == 1 or parts[1].lower() in ("help", "帮助"):
            yield event.plain_result(_usage())
            return

        sub = parts[1].lower()

        # ---- new code-execution sub-commands ----
        if sub in ("compute", "draw", "analyze") and len(parts) >= 3:
            code = parts[2]
            yield await self._run_code(
                event, code, mode=sub, output_dir=self.output_dir
            )
            return

        # ---- old-style JSON-based commands ----
        if len(parts) < 4:
            yield event.plain_result("参数不完整。\n\n" + _usage())
            return

        family = sub
        tool = parts[2].lower()
        payload = parts[3]

        try:
            kwargs = _parse_payload(payload)
            if family == "plot":
                fn = IMAGE_TOOLS.get(tool)
                if fn is None:
                    raise ValueError(f"未知绘图工具：{tool}")
                image = fn(**kwargs)
                yield self._image_result(event, self._save_image(image))
                return

            fn = TEXT_TOOLS.get((family, tool))
            if fn is None:
                raise ValueError(f"未知工具：{family} {tool}")
            yield event.plain_result(_json_dumps(fn(**kwargs)))
        except Exception as exc:
            yield event.plain_result(f"Fermat 执行失败：{exc}")

    # ------------------------------------------------------------------
    # Core execution helper (shared by command + LLM tools)
    # ------------------------------------------------------------------
    async def _run_code(
        self,
        event: AstrMessageEvent,
        code: str,
        mode: str = "compute",
        output_dir: Path | None = None,
    ):
        """Run user code and yield text + image results."""
        out = execute(code, output_dir or self.output_dir)

        lines: list[str] = []
        if out["error"]:
            lines.append(f"执行出错：\n{out['error']}")
        elif out["text"]:
            lines.append(out["text"])
        else:
            lines.append("(无文本输出)")

        if out["images"]:
            yield event.plain_result("\n".join(lines))
            for _, img in out["images"]:
                yield self._image_result(event, self._save_image(img))
        else:
            yield event.plain_result("\n".join(lines))

    # ==================================================================
    # Helpers for context-friendly output
    # ==================================================================
    MAX_CONTEXT_CHARS = 300  # never return more than this to LLM

    def _summarize(self, text: str) -> str:
        t = text.strip()
        if len(t) <= self.MAX_CONTEXT_CHARS:
            return t
        return t[: self.MAX_CONTEXT_CHARS] + "\n...(已截断)"

    def _img_summary(self, what: str) -> str:
        return f"[已生成图像: {what}]"

    def _compute_summary(self, description: str, result: str) -> str:
        summary = f"[{description}] {self._summarize(result)}"
        return summary

    # ==================================================================
    # LLM Tools — context-friendly summaries
    # ==================================================================

    @filter.llm_tool(name="fermat_compute")
    async def fermat_compute(self, event: AstrMessageEvent, code: str):
        """Execute Python code for mathematical / numerical computation.

        Libraries pre-imported: np (numpy), sp (scipy), sympy.
        Use print() to output results.

        Args:
            code: Python code string. Use np.sin(), sympy.diff(), etc.
        """
        out = execute(code, self.output_dir)
        if out["error"]:
            return f"计算失败：{out['error'].split(chr(10))[-2]}"  # last meaningful line
        result = out["text"] or "(无输出)"
        return self._compute_summary("科学计算完成", result)

    @filter.llm_tool(name="fermat_draw")
    async def fermat_draw(self, event: AstrMessageEvent, code: str):
        """Draw with Matplotlib. Use for creative/free-form plots (cat, custom chart, etc).

        Libraries pre-imported: plt (matplotlib.pyplot), np (numpy).
        DO NOT call plt.show() or plt.savefig() — figure captured automatically.

        Args:
            code: Matplotlib drawing code.
        """
        out = execute(code, self.output_dir)
        if out["error"]:
            yield event.plain_result(f"绘图失败：{out['error'].split(chr(10))[-2]}")
            return
        if out["images"]:
            items = [Comp.Plain(self._img_summary("自定义 Matplotlib 图形"))]
            for _, img in out["images"]:
                items.append(Comp.Image.fromFileSystem(self._save_image(img)))
            yield event.chain_result(items)
        else:
            yield event.plain_result("未生成图像，请检查代码是否创建了图形。")

    @filter.llm_tool(name="fermat_analyze")
    async def fermat_analyze(self, event: AstrMessageEvent, code: str):
        """Execute Python code for data analysis with Pandas/NumPy/SciPy.

        Libraries pre-imported: pd (pandas), np (numpy), sp (scipy).

        Args:
            code: Pandas data analysis code.
        """
        out = execute(code, self.output_dir)
        if out["error"]:
            return f"数据分析失败：{out['error'].split(chr(10))[-2]}"
        result = out["text"] or "(无输出)"
        return self._compute_summary("数据分析完成", result)

    # ==================================================================
    # High-frequency quick tools (no code writing needed)
    # ==================================================================

    @filter.llm_tool(name="fermat_plot_function")
    async def fermat_plot_function(
        self, event: AstrMessageEvent,
        expressions: str,
        x_min: float = -10.0,
        x_max: float = 10.0,
        title: str = "",
        grid: bool = True,
    ):
        """Plot mathematical functions. Use for "画个二次函数", "plot sin(x)", etc.

        This is the SIMPLEST drawing tool — just pass the expression string.

        Args:
            expressions: e.g. "x**2" or "x**2, sin(x), cos(x)" for multiple.
            x_min: Left bound (default -10).
            x_max: Right bound (default 10).
            title: Plot title (optional).
            grid: Show grid (default True).
        """
        funcs = [f.strip() for f in expressions.split(",") if f.strip()]
        if not funcs:
            yield event.plain_result("请提供至少一个函数表达式，如 x**2 或 sin(x)")
            return

        code_lines = [
            "import numpy as np",
            "x = np.linspace(x_min, x_max, 500)",
        ]
        for f in funcs:
            code_lines.append(f"plt.plot(x, {f}, linewidth=2, label='{f}')")
        if len(funcs) > 1:
            code_lines.append("plt.legend()")
        code_lines.append("plt.title(title) if title else plt.title(', '.join(funcs))")
        code_lines.append("plt.xlabel('x'); plt.ylabel('y')")
        if grid:
            code_lines.append("plt.grid(True, alpha=0.3)")

        code = "\n".join(code_lines)
        code = code.replace("x_min", str(x_min)).replace("x_max", str(x_max))
        code = f"title = {repr(title)}\n" + code

        out = execute(code, self.output_dir)
        if out["error"]:
            yield event.plain_result(f"绘图失败：{out['error'].split(chr(10))[-2]}")
            return
        if out["images"]:
            what = ", ".join(funcs)
            rng = f"[{x_min}, {x_max}]"
            items = [Comp.Plain(self._img_summary(f"函数 {what} 在 {rng}"))]
            for _, img in out["images"]:
                items.append(Comp.Image.fromFileSystem(self._save_image(img)))
            yield event.chain_result(items)
        else:
            yield event.plain_result("未生成图像")

    @filter.llm_tool(name="fermat_quick_compute")
    async def fermat_quick_compute(self, event: AstrMessageEvent, expression: str):
        """Evaluate a math expression numerically.

        Args:
            expression: e.g. "np.sin(np.pi/2)" or "2+3*5"
        """
        out = execute(f"result = {expression}\nprint(result)", self.output_dir)
        if out["error"]:
            return f"计算失败：{out['error'].split(chr(10))[-2]}"
        result = out["text"].strip() if out["text"] else "(无输出)"
        return self._summarize(result)

    # ==================================================================
    # Legacy LLM tools (kept for backward compatibility)
    # ==================================================================

    @filter.llm_tool(name="fermat_sympy_algebra")
    async def fermat_sympy_algebra(self, event: AstrMessageEvent, operation: str, expr: str, syms: Any = None):
        del event
        return self._summarize(algebra_operation(operation=operation, expr=expr, syms=syms))

    @filter.llm_tool(name="fermat_sympy_calculus")
    async def fermat_sympy_calculus(
        self, event: AstrMessageEvent, operation: str, expr: str, sym: str,
        n: int = 1, lower: Any = None, upper: Any = None,
        point: Any = 0, direction: str = "+", series_n: int = 6,
    ):
        del event
        return self._summarize(calculus_operation(
            operation, expr, sym, n, lower, upper, point, direction, series_n))

    @filter.llm_tool(name="fermat_sympy_equation")
    async def fermat_sympy_equation(self, event: AstrMessageEvent, operation: str, equations: Any, symbols: Any = None):
        del event
        return self._summarize(equation_operation(
            operation=operation, equations=equations, symbols=symbols))

    @filter.llm_tool(name="fermat_sympy_matrix")
    async def fermat_sympy_matrix(self, event: AstrMessageEvent, operation: str, data: Any):
        del event
        return self._summarize(_json_dumps(matrix_operation(operation=operation, data=data)))

    @filter.llm_tool(name="fermat_numpy")
    async def fermat_numpy(self, event: AstrMessageEvent, operation: str, a: Any = None, b: Any = None):
        del event
        return self._summarize(_json_dumps(numerical_operation(
            operation=operation, a=a, b=b)))

    @filter.llm_tool(name="fermat_plot_equation")
    async def fermat_plot_equation(
        self, event: AstrMessageEvent, equations: Any, x_min: float = -10.0,
        x_max: float = 10.0, title: str = "Equation Plot",
    ):
        image = eqn_chart(equations=equations, x_min=x_min, x_max=x_max, title=title)
        eqs = equations if isinstance(equations, str) else ", ".join(str(e) for e in equations)
        items = [
            Comp.Plain(self._img_summary(f"方程 {eqs} 在 [{x_min}, {x_max}]")),
            Comp.Image.fromFileSystem(self._save_image(image)),
        ]
        yield event.chain_result(items)

    async def terminate(self):
        pass
