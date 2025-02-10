# QuantLang

QuantLang is a Python-based domain-specific language (DSL) designed for evaluating financial portfolios and calculating various financial metrics—most notably, option Greeks (such as vega, delta, and theta). The project builds a custom abstract syntax tree (AST) from financial expressions and uses it to evaluate portfolio values, apply currency conversion, and aggregate Greek values. It also features visualization tools to help you analyze your portfolio’s risk metrics.

## Features

- **Expression Tokenization & Parsing**  
  - Tokenizes user-entered financial expressions (e.g., `"vega((0.4*aapl + 0.6*nvda) / usdsgd)"`).
  - Constructs an AST using custom node classes to represent assets, operations, numeraires, and Greek functions.
  
- **Portfolio Evaluation**  
  - Evaluates expressions by fetching market prices for assets, applying weights, and converting currencies using foreign exchange (FX) rates.
  - Uses static mappings for market prices, FX rates, and option Greeks (with the potential to extend these to live APIs).

- **Option Greek Calculation**  
  - Recursively calculates option Greeks by traversing the AST.
  - Aggregates individual Greek contributions (e.g., from assets weighted in a portfolio) to compute portfolio-level Greeks.

- **Data Integration & Visualization**  
  - Loads options and Greeks data from Parquet files.
  - Provides functions to filter options data (e.g., selecting near-ATM or ITM call options with maturities close to a target value).
  - Generates visualizations (using matplotlib) of Greek metrics (like Vega vs. Strike Prices and Theta vs. Time to Maturity) and embeds plots in HTML output.

- **Interactive Command-Line Interface**  
  - An interactive `main()` function accepts user expressions, displays debug information (including the AST), evaluates the expression, and outputs both numeric results and HTML-based plots.

- **Debugging & Traceability**  
  - Extensive debug logging to trace tokenization, AST construction, and evaluation steps.

## Project Structure
```
├── DataSource.ipynb
├── NLP.ipynb
├── QuantLang.ipynb
├── README.md
├── Test.ipynb
├── data
│   ├── aapl_momentum.png
│   ├── greeks_data.parquet
│   ├── options_data.parquet
│   └── quantlang_portfolio_report.html
└── modules
    ├── Option.py
    ├── chebyshev_interpolator.py
    ├── crank_nicolson.py
    ├── dq_plus.py
    ├── quadrature_nodes.py
    └── utils.py
```

