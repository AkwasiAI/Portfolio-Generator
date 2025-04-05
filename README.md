# Comprehensive Portfolio Generator

An AI-powered investment portfolio generator that creates detailed, data-driven portfolio reports with up-to-date market research.

## Features

- Generates comprehensive investment portfolio reports up to 13,000 words
- Includes executive summary with structured asset allocation table
- Integrates real-time market data via web searches (Perplexity API)
- Uses OpenAI's o3-mini model with high reasoning effort for content generation
- Covers multiple market sectors: global trade, energy, commodities, shipping
- Provides detailed asset analysis with allocation recommendations

## Requirements

- Python 3.8+
- OpenAI API key
- Perplexity API key

## Installation

1. Clone this repository
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
```

## Usage

Run the portfolio generator:

```
python comprehensive_portfolio_generator.py
```

The script will:
1. Run web searches to gather current market data
2. Generate each section of the portfolio report
3. Save the report as markdown and portfolio data as JSON
4. Output summary statistics about the generated portfolio

## Output

- `output/comprehensive_portfolio_report.md` - The detailed portfolio report
- `output/comprehensive_portfolio_data.json` - Structured JSON data of the portfolio

## Report Sections

1. Executive Summary (with portfolio table)
2. Global Trade & Economy
3. Energy Markets
4. Commodities
5. Shipping Sectors
6. Asset Analysis
7. Performance Benchmarking
8. Risk Assessment
9. Conclusion
