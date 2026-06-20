# AstrBot Fermat

AstrBot Fermat 是从 [abhiphile/fermat-mcp](https://github.com/abhiphile/fermat-mcp) 重构而来的 AstrBot 插件，用于在聊天中进行数学计算和绘图。

## 功能

- SymPy 符号计算：代数化简、展开、因式分解、求导、积分、极限、级数、方程求解、符号矩阵。
- NumPy 数值计算：数组运算、统计量、线性代数、特征值、SVD、线性方程组。
- Matplotlib 绘图：函数图像、折线图、散点图、柱状图、茎叶图、堆叠图。
- 支持手动命令和 AstrBot LLM Tool 自动调用。

## 安装

将本目录放入 AstrBot 的 `data/plugins/` 目录，或发布到插件仓库后通过插件市场安装。AstrBot 会根据 `requirements.txt` 安装依赖。

## 手动命令

命令参数使用 JSON 对象，避免矩阵、数组和表达式被空格拆开。

```text
/fermat sympy algebra {"operation":"simplify","expr":"(x+1)**2-x**2"}
/fermat sympy calculus {"operation":"diff","expr":"x**3","sym":"x"}
/fermat sympy equation {"operation":"solve","equations":"x**2-1","symbols":"x"}
/fermat sympy matrix {"operation":"det","data":"1 2; 3 4"}
/fermat numpy numerical {"operation":"mean","a":[1,2,3]}
/fermat plot equation {"equations":["x**2","sin(x)"],"x_min":-5,"x_max":5}
```

可用入口：

- `sympy algebra|calculus|equation|matrix`
- `numpy numerical|matlib`
- `plot bar|scatter|chart|stem|stack|equation`

## LLM Tools

插件注册了以下工具，启用函数调用后模型可自动使用：

- `fermat_sympy_algebra`
- `fermat_sympy_calculus`
- `fermat_sympy_equation`
- `fermat_sympy_matrix`
- `fermat_numpy`
- `fermat_plot_equation`

## 许可

核心数学功能来自 `abhiphile/fermat-mcp`，原项目使用 MIT License。本插件保留原 MIT License。
