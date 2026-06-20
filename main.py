import json
import uuid
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

from astrbot.api.event import AstrMessageEvent, filter
import astrbot.api.message_components as Comp
from astrbot.api.star import Context, Star, register

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


TEXT_TOOLS: dict[tuple[str, str], Callable[..., Any]] = {
    ("numpy", "numerical"): numerical_operation,
    ("numpy", "matlib"): matlib_operation,
    ("sympy", "algebra"): algebra_operation,
    ("sympy", "calculus"): calculus_operation,
    ("sympy", "equation"): equation_operation,
    ("sympy", "matrix"): matrix_operation,
}

IMAGE_TOOLS: dict[str, Callable[..., Any]] = {
    "bar": plot_barchart,
    "scatter": plot_scatter,
    "chart": plot_chart,
    "stem": plot_stem,
    "stack": plot_stack,
    "equation": eqn_chart,
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _parse_payload(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON 参数必须是对象，例如 {\"operation\": \"simplify\", \"expr\": \"x+x\"}")
    return value


def _usage() -> str:
    return (
        "Fermat 数学插件\n"
        "用法：\n"
        "/fermat sympy algebra {\"operation\":\"simplify\",\"expr\":\"(x+1)**2-x**2\"}\n"
        "/fermat sympy calculus {\"operation\":\"diff\",\"expr\":\"x**3\",\"sym\":\"x\"}\n"
        "/fermat sympy equation {\"operation\":\"solve\",\"equations\":\"x**2-1\",\"symbols\":\"x\"}\n"
        "/fermat sympy matrix {\"operation\":\"det\",\"data\":\"1 2; 3 4\"}\n"
        "/fermat numpy numerical {\"operation\":\"mean\",\"a\":[1,2,3]}\n"
        "/fermat plot equation {\"equations\":[\"x**2\",\"sin(x)\"],\"x_min\":-5,\"x_max\":5}\n"
        "可用模块：sympy/algebra|calculus|equation|matrix，numpy/numerical|matlib，plot/bar|scatter|chart|stem|stack|equation"
    )


@register(
    "astrbot_plugin_fermat",
    "OpenAI Codex + abhiphile",
    "数学计算与绘图插件，重构自 fermat-mcp。",
    "0.1.0",
)
class FermatPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.output_dir = Path(__file__).resolve().parent / "generated"
        self.output_dir.mkdir(exist_ok=True)

    def _save_image(self, image: Any) -> str:
        image_format = getattr(image, "format", "png") or "png"
        path = self.output_dir / f"fermat_{uuid.uuid4().hex}.{image_format}"
        path.write_bytes(image.data)
        return str(path)

    def _image_result(self, event: AstrMessageEvent, path: str):
        return event.chain_result([Comp.Image.fromFileSystem(path)])

    @filter.command("fermat")
    async def fermat(self, event: AstrMessageEvent):
        """Run Fermat math and plotting tools."""
        message = event.message_str.strip()
        parts = message.split(maxsplit=3)

        if len(parts) == 1 or (len(parts) > 1 and parts[1].lower() in {"help", "帮助"}):
            yield event.plain_result(_usage())
            return

        if len(parts) < 4:
            yield event.plain_result("参数不完整。\n\n" + _usage())
            return

        family = parts[1].lower()
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

    @filter.llm_tool(name="fermat_sympy_algebra")
    async def fermat_sympy_algebra(self, event: AstrMessageEvent, operation: str, expr: str, syms: Any = None):
        """Do symbolic algebra with SymPy.

        Args:
            operation(string): One of simplify, expand, factor, collect.
            expr(string): SymPy expression string.
            syms(object): Symbol or symbols used by collect.
        """
        del event
        return algebra_operation(operation=operation, expr=expr, syms=syms)

    @filter.llm_tool(name="fermat_sympy_calculus")
    async def fermat_sympy_calculus(
        self,
        event: AstrMessageEvent,
        operation: str,
        expr: str,
        sym: str,
        n: int = 1,
        lower: Any = None,
        upper: Any = None,
        point: Any = 0,
        direction: str = "+",
        series_n: int = 6,
    ):
        """Do calculus with SymPy.

        Args:
            operation(string): One of diff, integrate, limit, series.
            expr(string): SymPy expression string.
            sym(string): Variable name.
            n(number): Derivative order.
            lower(object): Lower bound for definite integrals.
            upper(object): Upper bound for definite integrals.
            point(object): Limit or series expansion point.
            direction(string): Limit direction, + or -.
            series_n(number): Number of series terms.
        """
        del event
        return calculus_operation(operation, expr, sym, n, lower, upper, point, direction, series_n)

    @filter.llm_tool(name="fermat_sympy_equation")
    async def fermat_sympy_equation(self, event: AstrMessageEvent, operation: str, equations: Any, symbols: Any = None):
        """Solve equations with SymPy.

        Args:
            operation(string): One of solve, solveset, linsolve, nonlinsolve.
            equations(object): Equation string or list of equation strings.
            symbols(object): Symbol or list of symbols to solve for.
        """
        del event
        return equation_operation(operation=operation, equations=equations, symbols=symbols)

    @filter.llm_tool(name="fermat_sympy_matrix")
    async def fermat_sympy_matrix(self, event: AstrMessageEvent, operation: str, data: Any):
        """Do symbolic matrix operations.

        Args:
            operation(string): One of create, det, inv, rref, eigenvals.
            data(object): Matrix data, such as "1 2; 3 4" or [[1, 2], [3, 4]].
        """
        del event
        return _json_dumps(matrix_operation(operation=operation, data=data))

    @filter.llm_tool(name="fermat_numpy")
    async def fermat_numpy(self, event: AstrMessageEvent, operation: str, a: Any = None, b: Any = None):
        """Do numerical NumPy operations.

        Args:
            operation(string): Operation such as add, mean, det, eig, solve, svd.
            a(object): First array, matrix, or scalar input.
            b(object): Second array, matrix, or scalar input.
        """
        del event
        return _json_dumps(numerical_operation(operation=operation, a=a, b=b))

    @filter.llm_tool(name="fermat_plot_equation")
    async def fermat_plot_equation(
        self,
        event: AstrMessageEvent,
        equations: Any,
        x_min: float = -10.0,
        x_max: float = 10.0,
        title: str = "Equation Plot",
    ):
        """Plot mathematical equations and send the image.

        Args:
            equations(object): Equation string or list of equation strings.
            x_min(number): Minimum x value.
            x_max(number): Maximum x value.
            title(string): Plot title.
        """
        image = eqn_chart(equations=equations, x_min=x_min, x_max=x_max, title=title)
        yield self._image_result(event, self._save_image(image))

    async def terminate(self):
        pass
