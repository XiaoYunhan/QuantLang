# QuantLang

QuantLang is a Python-based domain-specific language (DSL) designed for evaluating financial portfolios and calculating various financial metrics, such as option Greeks. The project leverages Yahoo Finance for fetching real-time and historical financial data.

## Features

- Tokenization and parsing of financial expressions
- Evaluation of financial expressions involving assets, operations, and numeraires
- Calculation of option Greeks (e.g., vega, delta, theta)
- Caching of foreign exchange (FX) rates and historical prices
- Debugging output for tracing the evaluation process

## Project Structure

- `QuantLang.ipynb`: The main Jupyter Notebook containing the implementation of QuantLang.

## Classes

### Node
A base class for all nodes in the abstract syntax tree (AST).

### WeightedAssetNode
Represents an asset with a specific weight.

### OperationNode
Represents an operation (e.g., '+', '-', '/') with child nodes.

### NumeraireNode
Represents a numeraire (e.g., a currency).

### FunctionNode
Represents a financial function (e.g., 'vega', 'delta') with an argument.

## Functions

### `tokenize(expression)`
Tokenizes a financial expression into a list of tokens.

### `parse(tokens)`
Parses a list of tokens into an AST.

### `evaluate(node)`
Evaluates an AST node.

### `calculate_greek(node, greek)`
Calculates the specified Greek for an AST node.

### `get_historical_price(asset, date)`
Fetches the historical price of an asset.

### `get_fx_rate(currency_pair)`
Fetches the real-time FX rate for a currency pair.

## Usage

1. Clone the repository.
2. Open `QuantLang.ipynb` in Jupyter Notebook.
3. Run the cells to initialize the classes and functions.
4. Use the `main()` function to interact with the Portfolio Calculator.

## Example

```python
# Example expression: Calculate the vega of a portfolio
expression = "vega((0.4*aapl + 0.6*nvda) / usdsgd)"
tokens = tokenize(expression)
ast = parse(tokens)
result = evaluate(ast)
print(f"Result: {result}")
```

## Acknowledgments
Yahoo Finance for financial data.