# Comprehensive Portfolio Generator

An AI-powered investment portfolio generator that creates detailed, data-driven portfolio reports with up-to-date market research.

## Features

- Generates comprehensive investment portfolio reports up to 13,000 words
- Includes executive summary with structured asset allocation table
- Integrates real-time market data via web searches (Perplexity API)
- Uses OpenAI's o3-mini model with high reasoning effort for content generation
- Covers multiple market sectors: global trade, energy, commodities, shipping
- Provides detailed asset analysis with allocation recommendations
- Automatically uploads reports and portfolio data to Firestore (optional)

## Requirements

- Python 3.8+
- OpenAI API key
- Perplexity API key
- Google Cloud credentials (optional, for Firestore integration)

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

4. (Optional) For Firestore integration, set up Google Cloud credentials:
   - Create a service account in Google Cloud Console with Firestore permissions
   - Download the service account key JSON file
   - Simply place the JSON file in the project root directory
      - The system will automatically detect and use it
      - Supported filenames: service account files or any with "hedgefundintelligence" or "credentials" in the name
   - Alternatively, you can manually set the environment variable:
```
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-service-account-key.json"
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
5. Upload the report and portfolio data to Firestore (automatically if credentials file is in project directory)

## Output

- `output/comprehensive_portfolio_report.md` - The detailed portfolio report
- `output/comprehensive_portfolio_data.json` - Structured JSON data of the portfolio

If Firestore integration is enabled, the reports and portfolio data will also be stored in your Firestore database under the 'portfolios' collection with their respective document types ('reports' and 'portfolio_weights'). The system automatically detects credential files in the project directory, so you don't need to set any environment variables manually.

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

## Firestore Integration

The Portfolio Generator now supports automatic uploading to Firestore with simplified credential handling:

1. **Automatic Credential Detection**: Place your Google Cloud service account JSON file in the project root directory, and the system will automatically find and use it.

2. **No Manual Environment Variables**: No need to set `GOOGLE_APPLICATION_CREDENTIALS` manually - the system handles it for you.

3. **Supported Credential Filenames**:
   - Any file matching `hedgefundintelligence*.json`
   - Files with `credential` in the name
   - Standard service account files like `service-account*.json`
   - Any JSON file containing service account key indicators

For additional management of your Firestore documents, a command-line tool is provided:

```
python portfolio_cli.py upload <filename> --type=<type> [--format=<format>] [--not-latest]
python portfolio_cli.py get-latest --type=<type> [--output=<output_file>]
python portfolio_cli.py list [--limit=<num>] [--type=<type>]
```

Where:
- `type` can be 'reports', 'portfolio_weights', or 'report_feedback'
- `format` can be 'markdown', 'json', or 'auto' (default)

Example usage:
```
# List recent portfolio documents
python portfolio_cli.py list --limit=5

# Download the latest report
python portfolio_cli.py get-latest --type=reports --output=latest_report.md
```
