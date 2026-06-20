"""Safe Python code execution engine for scientific computing.

Pre-imports numpy, scipy, sympy, pandas, matplotlib so LLM-generated
code can use them directly without import boilerplate.
"""

import io
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

from .image import Image


# ---- Chinese font setup for Linux ----
def _setup_chinese_font():
    """Try common Chinese fonts, fall back gracefully."""
    candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Source Han Sans SC",
        "AR PL UMing CN",
        "AR PL UKai CN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    # No Chinese font found — use sans-serif with minus fix
    plt.rcParams["axes.unicode_minus"] = False
    return None


_chinese_font = _setup_chinese_font()

# ---- Common numpy functions exposed directly (so sin(x) works, not np.sin(x)) ----
_NP_DIRECT = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs,
    "pi": np.pi, "e": np.e,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "degrees": np.degrees, "radians": np.radians,
    "ceil": np.ceil, "floor": np.floor, "round": np.round,
    "linspace": np.linspace, "arange": np.arange,
    "array": np.array, "zeros": np.zeros, "ones": np.ones,
}


# Libraries exposed to user code (pre-imported, no import needed)
EXPORTS: dict[str, Any] = {
    "np": np,
    "plt": plt,
    "pd": pd,
    "sys": sys,
    **_NP_DIRECT,
}


def _lazy_import(name: str, package: str) -> Any:
    try:
        mod = __import__(package, fromlist=["*"])
        EXPORTS[name] = mod
        return mod
    except ImportError:
        return None


def _get_sympy() -> Any:
    if "sympy" in EXPORTS:
        return EXPORTS["sympy"]
    import sympy
    EXPORTS["sympy"] = sympy
    return sympy


def _get_scipy() -> Any:
    if "sp" in EXPORTS:
        return EXPORTS["sp"]
    try:
        import scipy
        import scipy.integrate
        import scipy.optimize
        import scipy.linalg
        import scipy.stats
        import scipy.signal
        import scipy.fft
        EXPORTS["sp"] = scipy
        EXPORTS["integrate"] = scipy.integrate
        EXPORTS["optimize"] = scipy.optimize
        EXPORTS["linalg"] = scipy.linalg
        EXPORTS["stats"] = scipy.stats
        EXPORTS["signal"] = scipy.signal
        EXPORTS["fft"] = scipy.fft
        return scipy
    except ImportError:
        return None


def execute(code: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Run user-supplied Python code in a pre-loaded scientific namespace.

    Returns a dict with:
        - text: captured stdout as string
        - images: list of (path, Image) for any matplotlib figures drawn
        - error: exception traceback if code crashed (None on success)
    """
    _get_sympy()
    _get_scipy()

    old_stdout = sys.stdout
    capture = io.StringIO()
    sys.stdout = capture

    result: dict[str, Any] = {"text": "", "images": [], "titles": [], "error": None}

    try:
        plt.close("all")

        exec(code, EXPORTS)

        for fig_num in plt.get_fignums():
            f = plt.figure(fig_num)
            if f.get_axes():
                # Extract titles from all axes
                titles = []
                for ax in f.get_axes():
                    t = ax.get_title()
                    if t:
                        titles.append(t)
                result["titles"].append("; ".join(titles) if titles else "")

                buf = io.BytesIO()
                f.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(f)
                buf.seek(0)
                img_path = str(output_dir / f"fermat_plot_{fig_num}.png") if output_dir else None
                result["images"].append(
                    (img_path, Image(data=buf.read(), format="png"))
                )

    except Exception as exc:
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    result["text"] = capture.getvalue().strip()
    return result
