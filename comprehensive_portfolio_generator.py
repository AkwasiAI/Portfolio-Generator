#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import re
from portfolio_generator.web_search import PerplexitySearch

# Define logging functions for better visibility
def log_error(message):
    print(f"\033[91m[ERROR] {message}\033[0m")
    
def log_warning(message):
    print(f"\033[93m[WARNING] {message}\033[0m")
    
def log_success(message):
    print(f"\033[92m[SUCCESS] {message}\033[0m")
    
def log_info(message):
    print(f"\033[94m[INFO] {message}\033[0m")

def format_search_results(search_results):
    """Format search results for use in prompts."""
    if not search_results:
        return ""
    
    # Filter results to only include those with actual content
    valid_results = [r for r in search_results 
                    if r.get("results") and len(r["results"]) > 0 and "content" in r["results"][0]]
    
    if not valid_results:
        log_warning("No valid search results to format - all results were empty or had errors")
        return ""
        
    formatted_text = "\n\nWeb Search Results (current as of 2025):\n"
    
    for i, result in enumerate(valid_results):
        query = result.get("query", "Unknown query")
        content = result["results"][0].get("content", "No content available")
        
        formatted_text += f"\n---Result {i+1}: {query}---\n{content}\n"
    
    log_info(f"Formatted {len(valid_results)} valid search results for use in prompts")
    return formatted_text

async def generate_section(client, section_name, system_prompt, user_prompt, search_results=None):
    """Generate a section of the investment portfolio report."""
    print(f"Generating {section_name}...")
    
    try:
        # Create messages for the API call
        messages = [
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add web search results if available
        if search_results and search_results.strip():
            messages.append({"role": "user", "content": "Here is the latest information from web searches:\n\n" + search_results})
        
        log_info(f"Generating section {section_name} using o3-mini model with high reasoning effort")
        response = client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="high"
        )
        
        # Get the content
        section_content = response.choices[0].message.content
        return section_content
    
    except Exception as e:
        error_msg = f"Error generating section {section_name}: {e}"
        print(f"\033[91m{error_msg}\033[0m")
        prompt_continue = input("Do you want to continue despite this error? (y/n): ")
        if prompt_continue.lower() != 'y':
            print("Exiting script due to generation error.")
            sys.exit(1)
        return f"## {section_name}\n\nError generating content: {e}\n\n"

async def generate_portfolio_json(client, assets_list, current_date, search_client=None, search_results=None):
    """Generate the structured JSON portfolio data."""
    system_prompt = """You are a data structuring assistant for Orasis Capital. 
Your task is to convert portfolio asset information into a structured JSON format.

Currently it is April 2025. Use this current date for all information.

You MUST respond with ONLY valid JSON, nothing else. No explanations, no other text, no code blocks, no backticks.

Be extremely precise in following the requested JSON structure and ensure all values add up correctly."""

    # Create a detailed prompt with the asset list
    assets_str = "\n".join([f"- {asset}" for asset in assets_list])
    
    # Create a template for the JSON structure
    json_template = {
        "status": "success",
        "data": {
            "report_date": current_date,
            "assets": [
                {
                    "asset_name": "Full asset name including ticker",
                    "category": "Asset category (Shipping Equity, Commodity, Bond, etc.)",
                    "region": "Geographic region",
                    "weight": "Numerical allocation percentage without % sign",
                    "horizon": "Time horizon (Short (1-3M), Medium (3-6M), Long (6-12M))",
                    "recommendation": "Buy/Sell/Hold plus Long/Short",
                    "rationale": "Brief 1-line rationale with key data point"
                }
            ],
            "summary": {
                "by_category": {
                    "Category1": "Sum of weights for this category",
                    "Category2": "Sum of weights for this category"
                },
                "by_region": {
                    "Region1": "Sum of weights for this region",
                    "Region2": "Sum of weights for this region"
                },
                "by_recommendation": {
                    "Recommendation1": "Sum of weights for this recommendation",
                    "Recommendation2": "Sum of weights for this recommendation"
                }
            },
            "references": [
                {
                    "id": "ref1",
                    "category": "Source category (Energy, Shipping, Economic, etc.)",
                    "author": "Author or Organization",
                    "title": "Publication title",
                    "publisher": "Publisher/Journal/Website",
                    "date": "Publication date (use 2024-2025 dates)",
                    "url": "URL if available"
                }
            ]
        }
    }
    
    # Convert the template to a string representation for the prompt
    json_template_str = json.dumps(json_template, indent=2)
    
    user_prompt = """Based on the following asset list, create a complete structured JSON object in the specified format.

It is currently April 2025. You must use the most recent data and references available up through 2025. Do not mention or acknowledge any knowledge cutoff dates.

Asset list:
""" + assets_str + """

Current date: """ + current_date + """

You MUST return ONLY valid JSON in the following structure, nothing else. No markdown code blocks, no backticks (```), no explanations:

""" + json_template_str + """

Ensure all assets add up to exactly 100% and that the JSON is valid. Include at least 25 reputable references across different categories from 2024-2025.
"""

    try:
        # Create messages for API call
        messages = [
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Skip web search integration in this specific function since we're doing it at a higher level
        log_info("Generating portfolio JSON data using o3-mini model with high reasoning effort")
        
        response = client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="high"
        )
        
        # Get the JSON content
        json_response = response.choices[0].message.content.strip()
        
        # Clean the response to ensure it's valid JSON
        # Remove any markdown code block indicators or extra text
        if json_response.startswith('```json'):
            json_response = json_response.split('```json', 1)[1]
        if json_response.startswith('```'):
            json_response = json_response.split('```', 1)[1]
        if '```' in json_response:
            json_response = json_response.split('```')[0]
        
        # Strip any leading/trailing whitespace or quotes
        json_response = json_response.strip('`\' \n"')
        
        # Validate the JSON before returning
        try:
            # Attempt to parse the JSON to validate it
            parsed_json = json.loads(json_response)
            return json.dumps(parsed_json, indent=2)  # Return properly formatted JSON
        except json.JSONDecodeError as json_err:
            print(f"JSON Parsing Error: {json_err}")
            # Fallback: try to extract JSON using regex if possible
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, json_response)
            if match:
                try:
                    extracted_json = match.group(0)
                    return json.dumps(json.loads(extracted_json), indent=2)
                except:
                    pass
            
            # If all else fails, return error
            return json.dumps({"status": "error", "message": f"JSON parsing error: {str(json_err)}"})
    except Exception as e:
        print(f"Error generating JSON data: {e}")
        return {"status": "error", "message": str(e)}

async def generate_investment_portfolio():
    """Generate a comprehensive investment portfolio report through multiple API calls."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\033[91mERROR: OPENAI_API_KEY environment variable is not set!\033[0m")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Initialize search client if available
    search_client = None
    
    # Make sure the environment variables are loaded correctly
    load_dotenv()  # Explicitly load .env file again to ensure variables are available
    
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    
    # Debug API key format (showing only first and last few characters for security)
    if perplexity_api_key:
        key_preview = f"{perplexity_api_key[:8]}...{perplexity_api_key[-5:]}" if len(perplexity_api_key) > 13 else "[key too short]"
        log_info(f"PERPLEXITY_API_KEY format: {key_preview} (length: {len(perplexity_api_key)})")
    # Check API key format - Perplexity keys usually start with 'pplx-'
    if perplexity_api_key and not perplexity_api_key.startswith("pplx-"):
        log_warning("Your Perplexity API key doesn't start with 'pplx-' which is the expected format")
        
    if perplexity_api_key:
        try:
            search_client = PerplexitySearch(api_key=perplexity_api_key)
            # Test the API key with a simple query
            test_query = "test query"
            log_info(f"Testing Perplexity API with query: {test_query}")
            
            test_result = await search_client.search([test_query])
            test_response = test_result[0]
            log_info(f"Test query response: {test_response}")
            
            # Check if we received actual content
            if test_response.get("results") and len(test_response["results"]) > 0:
                log_success("Perplexity API key validated successfully.")
            elif "error" in test_response:
                log_error("Perplexity API key is invalid or returned an error.")
                log_error(f"Error details: {test_response.get('error', 'Unknown error')}")
                prompt_continue = input("Do you want to continue without web search functionality? (y/n): ")
                if prompt_continue.lower() != 'y':
                    print("Exiting script. Please check your PERPLEXITY_API_KEY and try again.")
                    sys.exit(1)
                search_client = None
            else:
                log_success("Perplexity API key appears to be working.")
        except Exception as e:
            log_error(f"Error initializing Perplexity search: {e}")
            log_error(f"Error type: {type(e).__name__}")
            import traceback
            log_error(f"Traceback: {traceback.format_exc()}")
            
            prompt_continue = input("Do you want to continue without web search functionality? (y/n): ")
            if prompt_continue.lower() != 'y':
                print("Exiting script. Please check your PERPLEXITY_API_KEY and try again.")
                sys.exit(1)
            search_client = None
    else:
        log_warning("PERPLEXITY_API_KEY not set. Web search disabled.")
        prompt_continue = input("Do you want to continue without web search functionality? (y/n): ")
        if prompt_continue.lower() != 'y':
            print("Exiting script. Please set your PERPLEXITY_API_KEY and try again.")
            sys.exit(1)
    
    # Use a fixed date in 2025 as the current date
    current_date = "April 4, 2025"
    
    # Start time for tracking runtime
    start_time = time.time()
    
    # Perform web searches upfront to have the data available for all API calls
    formatted_search_results = ""
    if search_client:
        try:
            log_info("Performing web searches for market data upfront...")
            
            # List of search queries covering different aspects of the portfolio
            # Expanded to at least 30 searches for more comprehensive data
            search_queries = [
                # Global Economy & Trade (7 queries)
                "current global trade metrics and trends 2025",
                "global GDP growth forecast by region 2025",
                "international trade volumes by commodity 2025",
                "emerging markets economic outlook 2025",
                "global inflation rates and impact on trade 2025",
                "China trade policy and import/export volumes 2025",
                "supply chain disruptions and logistics trends 2025",
                
                # Shipping Sector - Detailed (10 queries)
                "container shipping rates and market trends 2025",
                "Baltic Dry Index latest values and forecasts 2025",
                "tanker shipping market rates and vessel utilization 2025",
                "VLCC spot rates and time charter rates 2025",
                "cape size vessel earnings and fleet growth 2025",
                "panamax and supramax market trends 2025",
                "LNG carrier market rates and orderbook 2025",
                "port congestion data and container throughput 2025",
                "shipping industry regulatory changes impact 2025",
                "IMO 2023 and emission regulations shipping impact 2025",
                
                # Energy Markets (6 queries)
                "crude oil price forecasts and inventory levels 2025",
                "natural gas market supply demand balance 2025",
                "LNG market growth and trade flows 2025",
                "renewable energy investment trends 2025",
                "energy transition impact on shipping 2025",
                "bunker fuel prices and trends 2025",
                
                # Commodities (6 queries)
                "iron ore market prices and production data 2025",
                "copper supply demand balance and price forecasts 2025",
                "aluminum market trends and inventory levels 2025",
                "agricultural commodities trade flows 2025",
                "grain production forecasts and shipping demand 2025",
                "commodity futures market positioning 2025",
                
                # Financial Markets (4 queries)
                "shipping company stock performance 2025",
                "global interest rates and bond market 2025",
                "currency exchange rates impact on shipping 2025",
                "shipping industry financing and debt levels 2025"
            ]
            
            log_info(f"Executing {len(search_queries)} web searches...")
            search_results = await search_client.search(search_queries)
            
            # Display detailed results of each web search for debugging
            for i, result in enumerate(search_results):
                result_str = str(result)
                
                # With the new API approach, check if the results list contains content
                if result.get("results") and len(result["results"]) > 0 and "content" in result["results"][0]:
                    content_preview = result["results"][0]["content"][:100]
                    log_success(f"Search {i+1} successful: '{result['query']}' → {content_preview}...")
                elif "error" in result:
                    log_error(f"Search {i+1} failed: {result.get('error', 'Unknown error')}")
                else:
                    log_warning(f"Search {i+1} returned empty or unexpected format: {result_str[:150]}")
                    
            # Check the quality of search results
            successful_searches = sum(1 for r in search_results if r.get("results") and len(r["results"]) > 0 and "content" in r["results"][0])
            failed_searches = len(search_results) - successful_searches
            
            if failed_searches == len(search_results):
                log_error("All search queries failed to return useful content.")
                prompt_continue = input("Continue without web search data? (y/n): ")
                if prompt_continue.lower() != 'y':
                    print("Exiting script. Please check your PERPLEXITY_API_KEY and try again.")
                    sys.exit(1)
            elif failed_searches > 0:
                log_warning(f"{failed_searches} out of {len(search_results)} searches failed to return useful content.")
            
            # Determine if we have usable search results
            has_errors = failed_searches > (len(search_results) / 2)  # More than half failed
            
            if has_errors:
                log_error("Found API authentication errors or empty results")
                error_sample = next((r for r in search_results if 'error' in str(r) or 'unauthorized' in str(r)), '')
                if error_sample:
                    log_error(f"Error sample: {error_sample}")
                else:
                    log_error("All search results were empty, indicating API key issues")
                prompt_continue = input("Continue without web search data? (y/n): ")
                if prompt_continue.lower() != 'y':
                    sys.exit(1)
                formatted_search_results = ""
                log_warning("No valid search results. Report will not include current data.")
            else:
                # Try to use any non-empty results
                formatted_search_results = format_search_results(search_results)
                if formatted_search_results:
                    log_success(f"Successfully formatted search results for use in prompts")
                else:
                    log_warning("No valid search results obtained. Report will not include current data.")
        except Exception as e:
            log_error(f"Exception during web search: {e}")
            log_error(f"Error type: {type(e).__name__}")
            import traceback
            log_error(f"Traceback: {traceback.format_exc()}")
            
            prompt_continue = input("Do you want to continue without web search functionality? (y/n): ")
            if prompt_continue.lower() != 'y':
                print("Exiting script. Please check your PERPLEXITY_API_KEY and try again.")
                sys.exit(1)
            formatted_search_results = ""
    
    # Base system prompt for all sections
    base_system_prompt = """You are a professional investment analyst at Orasis Capital, a hedge fund specializing in global macro and trade-related assets.
Your task is to create detailed investment portfolio analysis with data-backed research and specific source citations.

IMPORTANT CONSTRAINTS:
1. The ENTIRE report must be NO MORE than 13,000 words total. Optimize your content accordingly.
2. You MUST include a comprehensive summary table in the Executive Summary section.
3. Ensure all assertions are backed by specific data points or sources.
4. Use current data from 2024-2025 where available."""

    # Dictionary to store all sections
    sections = {}
    
    # 1. Generate Executive Summary
    log_info("Generating executive summary section...")
    exec_summary_prompt = f"""Generate an executive summary for the investment portfolio report.

Include current date ({current_date}) and the title format specified previously.
Summarize the key findings, market outlook, and high-level portfolio strategy.
Keep it clear, concise, and data-driven with specific metrics.

CRITICAL REQUIREMENT: You MUST include a comprehensive summary table displaying ALL portfolio positions (strictly limited to 10-15 total positions).
This table MUST be properly formatted in markdown and include columns for:
- Asset/Ticker
- Position Type (Long/Short)
- Allocation % (must sum to 100%)
- Time Horizon
- Confidence Level

Remember that the entire report must not exceed 13,000 words total. This summary should be concise but comprehensive.

After the table, include a brief overview of asset allocations by category (shipping, commodities, energy, etc.)."""
    
    sections["executive_summary"] = await generate_section(
        client, "Executive Summary", base_system_prompt, exec_summary_prompt, search_results=formatted_search_results
    )
    
    # 2. Generate Global Trade & Economy section
    global_economy_prompt = """Write a concise but comprehensive analysis (600-700 words) of Global Trade & Economy as part of a macroeconomic outlook section.
Include:
- Regional breakdowns and economic indicators with specific figures
- GDP growth projections by region with exact percentages
- Trade flow statistics with exact volumes and year-over-year changes
- Container throughput at major ports with specific TEU figures
- Supply chain metrics and logistics indicators
- Currency valuations and impacts on trade relationships
- Trade agreements and policy changes with implementation timelines
- Inflation rates across major economies with comparisons

Format in markdown starting with:
## Macroeconomic & Industry Outlook
### Global Trade & Economy

Include 5-7 specific sources (e.g., IMF, World Bank, WTO, UNCTAD, economic research firms, central banks) with publication dates.
Every assertion should be backed by data or a referenced source.

NOTE: Keep this section concise to ensure the entire report remains under the 13,000 word limit.
"""
    
    sections["global_economy"] = await generate_section(
        client, "Global Trade & Economy", base_system_prompt, global_economy_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Global Trade & Economy")
    
    # 3. Generate Energy Markets section
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Energy Markets")
    energy_markets_prompt = """Write a concise but informative analysis (500-600 words) of Energy Markets as part of a macroeconomic outlook section.
Include:
- Oil markets: supply/demand balance with specific production figures, inventory levels, and price projections
- Natural Gas & LNG: capacity expansions with exact volumes, trade routes, and pricing dynamics
- Renewable Energy transition impacts with adoption rates and investment figures
- Energy infrastructure developments with capacity and timeline data
- OPEC+ and non-OPEC production quotas and compliance rates
- Refining margins and utilization rates across regions

Format in markdown starting with:
### Energy Markets

Include 4-5 specific sources with publication dates.
Every assertion should be backed by data or a referenced source.

NOTE: Keep this section concise to ensure the entire report remains under the 13,000 word limit.
"""
    
    sections["energy_markets"] = await generate_section(
        client, "Energy Markets", base_system_prompt, energy_markets_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Energy Markets")
    
    # 4. Generate Commodities section
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Commodities")
    commodities_prompt = """Write a concise but informative analysis (500-600 words) of Commodities Markets as part of a macroeconomic outlook section.
Include:
- Metals: supply/demand fundamentals for copper, iron ore, aluminum with production figures and inventory levels
- Agricultural: crop reports, weather impacts, inventory-to-use ratios with specific figures
- Supply chain dynamics and infrastructure constraints with quantitative impacts
- Futures market positioning and price forecasts with technical levels
- Industrial demand trends by region with consumption metrics
- Production costs and margin analysis across commodity sectors

Format in markdown starting with:
### Commodities

Include 4-5 specific sources (e.g., USDA, LME, SGX, commodity research firms, production reports) with publication dates.
Every assertion should be backed by data or a referenced source.

NOTE: Keep this section concise to ensure the entire report remains under the 13,000 word limit.
"""
    
    sections["commodities"] = await generate_section(
        client, "Commodities", base_system_prompt, commodities_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Commodities")
    
    # 5. Generate Shipping Sectors section
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Shipping Sectors")
    shipping_prompt = """Write a concise but informative analysis (700-800 words) of Shipping Sectors as part of a macroeconomic outlook section.
Include:
- Tankers: fleet growth percentages, orderbook trends, ton-mile demand with specific figures
- Dry Bulk: BDI analysis with specific index levels, vessel categories performance, and spot/time charter rates
- Containers: TEU capacity, port congestion metrics, charter rates with specific USD/day figures
- LNG carriers: liquefaction capacity growth, vessel utilization rates, market rates
- Fleet age profiles and scrapping rates across sectors
- Regulatory impacts (IMO 2023, EEXI, CII) with compliance costs
- Regional trade flow shifts with specific route data

Format in markdown starting with:
### Shipping Sectors

Include 5-6 specific sources (e.g., Clarksons, Drewry, Alphaliner, Baltic Exchange, ship brokers, shipping companies) with publication dates.
Every assertion should be backed by data or a referenced source.

NOTE: Keep this section concise to ensure the entire report remains under the 13,000 word limit.
"""
    
    sections["shipping"] = await generate_section(
        client, "Shipping Sectors", base_system_prompt, shipping_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Shipping Sectors")
    
    # 6. Generate Portfolio Recommendations for 12 assets
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Portfolio Recommendations")
    # First, generate a list of 12 diverse assets across asset classes
    asset_prompt = """Create a list of 12 diverse investment assets that would be suitable for a trade-focused multi-asset portfolio.
Include a mix of:
- Shipping equities (tankers, dry bulk, containers, LNG)
- Energy equities and ETFs
- Commodity producers and ETFs
- Bonds and credit instruments
- Agricultural assets
- Infrastructure assets related to global trade

For each asset, provide:
1. Full name with ticker
2. Asset class/category
3. Geographic focus
4. A key data point or metric justifying its inclusion

Format as a simple list with one asset per line, but include all the information above for each asset.
"""
    
    asset_list_raw = await generate_section(
        client, "Asset List", base_system_prompt, asset_prompt, search_results=formatted_search_results
    )
    log_success(f"Generated asset list for section {current_section}/{total_sections}")
    
    # Parse the asset list into individual assets
    asset_lines = [line.strip() for line in asset_list_raw.split('\n') if line.strip()]
    asset_list = [line for line in asset_lines if not line.startswith('#') and not line.startswith('Asset List')]
    
    # Now generate detailed analysis for each asset (limit to 5 at a time to manage complexity)
    total_assets = len(asset_list)
    log_info(f"Preparing to generate analyses for {total_assets} assets")
    portfolio_items = []
    
    # Process assets in batches of 4
    for i in range(0, len(asset_list), 4):
        batch = asset_list[i:i+4]
        batch_prompts = []
        
        for j, asset in enumerate(batch):
            current_asset_num = i + j + 1
            log_info(f"Preparing asset analysis {current_asset_num}/{total_assets}: {asset[:50]}...")
            asset_prompt = f"""Write a concise but comprehensive analysis (300-400 words) for the following asset as part of an investment portfolio:

{asset}

Include:
- Complete company/instrument background
- Detailed category description and market position
- Geographic exposure and regional dynamics
- Long/short positioning recommendation with specific entry/exit criteria
- Weight (percentage allocation) with justification
- Investment time horizon with milestone triggers
- Confidence level with supporting evidence
- Comprehensive data-backed rationale with multiple metrics
- Competitor analysis and relative value assessment
- Historical performance analysis
- Technical analysis indicators
- Valuation metrics compared to sector averages

Format in markdown starting with a clear header for the asset name.
Include 3-4 specific sources relevant to this asset with publication dates.
Every assertion should be backed by data or a referenced source.

NOTE: Please keep your analysis BRIEF but COMPREHENSIVE to ensure the entire report remains under the 13,000 word limit.
"""
            batch_prompts.append(asset_prompt)
        
        # Run the prompts in parallel
        log_info(f"Generating analyses for assets {i+1}-{min(i+len(batch), total_assets)} of {total_assets}...")
        tasks = []
        for j, prompt in enumerate(batch_prompts):
            current_asset_num = i + j + 1
            tasks.append(generate_section(
                client, 
                f"Asset Analysis {current_asset_num}/{total_assets}", 
                base_system_prompt, 
                prompt, 
                search_results=formatted_search_results
            ))
        batch_results = await asyncio.gather(*tasks)
        
        log_success(f"Completed assets {i+1}-{min(i+len(batch), total_assets)} of {total_assets}")
        portfolio_items.extend(batch_results)
    
    # Join all portfolio items
    sections["portfolio_items"] = "\n\n## Portfolio Positioning & Rationale\n\n" + "\n\n".join(portfolio_items)
    log_success(f"Completed section {current_section}/{total_sections}: Portfolio Items")
    
    # 7. Generate Performance Benchmarking
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Performance Benchmarking")
    benchmarking_prompt = """Write a detailed Performance Benchmarking section (500+ words) for an investment portfolio.
Include:
- Detailed comparison to prior allocations with performance metrics
- Attribution analysis by sector and asset class with specific figures
- Risk-adjusted return calculations (Sharpe ratios, Sortino ratios, etc.)
- Benchmark comparisons (S&P 500, MSCI World, commodity indices, etc.)
- Performance during specific market regimes (high inflation, dollar strength, etc.)
- Factor attribution (value, momentum, quality, etc.)

Format in markdown starting with:
## Performance Benchmarking

Include at least 5-7 specific sources with publication dates.
Every assertion should be backed by data or a referenced source.
"""
    
    sections["benchmarking"] = await generate_section(
        client, "Performance Benchmarking", base_system_prompt, benchmarking_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Benchmarking")
    
    # 8. Generate Risk Assessment
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Risk Assessment")
    risk_prompt = """Write a detailed Risk Assessment & Monitoring Guidelines section (1000+ words) for an investment portfolio.
Include:
- Detailed key risk factors by asset and overall portfolio
- VaR and stress test scenarios with specific loss potentials
- Correlation analysis between positions with correlation coefficients
- Monitoring framework with specific metrics and thresholds
- Hedging strategies for key risk factors
- Liquidity risk assessment by asset class
- Concentration risk analysis
- Regulatory and compliance risks

Format in markdown starting with:
## Risk Assessment & Monitoring Guidelines

Include at least 5-7 specific sources with publication dates.
Every assertion should be backed by data or a referenced source.
"""
    
    sections["risk_assessment"] = await generate_section(
        client, "Risk Assessment", base_system_prompt, risk_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Risk Assessment")
    
    # Generate Portfolio Items Section
    log_info("Generating portfolio items section...")
    portfolio_prompt = """Generate the detailed portfolio positions section of the report.

STRICTLY LIMIT to exactly 10-15 investment positions TOTAL (mix of long/short) with detailed rationale for each.
These must be EXACTLY THE SAME positions as shown in the Executive Summary table.
Use specific asset names/tickers and ensure target allocation percentages add to exactly 100%.

For each position provide:
- Asset names/tickers
- Long/short positioning
- Target allocation percentages
- Investment time horizon (specific months/quarters)
- Confidence level (high/moderate/low) with justification
- Data-backed rationale with specific numbers
- Clear relation to the current market conditions

Organize by asset category and provide a clear explanation of how each aligns with the overall strategy.
Do not add any positions beyond the 10-15 shown in the Executive Summary table.
"""
    sections["portfolio_items"] = await generate_section(
        client, "Portfolio Positions", base_system_prompt, portfolio_prompt, search_results=formatted_search_results
    )
    
    # 9. Generate Summary Table and Conclusion
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: Conclusion")
    conclusion_prompt = """Write a concise Conclusion section with a comprehensive summary table of all portfolio recommendations.
The table should include:
- Asset name/ticker
- Category
- Region
- Position (Long/Short)
- Target allocation (%)
- Time horizon
- Confidence level
- Key rationale

Format in markdown starting with:
## Conclusion and Summary Recommendations

Follow the conclusion text with a properly formatted markdown table of all positions.

Include 3-5 specific sources with publication dates.
"""
    
    sections["conclusion"] = await generate_section(
        client, "Conclusion and Summary", base_system_prompt, conclusion_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: Conclusion")
    
    # 10. Generate References
    current_section += 1
    log_info(f"Generating section {current_section}/{total_sections}: References")
    references_prompt = """Create a comprehensive References section with at least 30 specific sources used throughout the report.
Categorize sources by sector (Energy, Shipping, Commodities, etc.).
Include:
- Research reports
- Regulatory filings
- Industry publications
- Consultant reports
- Company presentations
- Economic data providers
- Academic papers

For each reference, include:
- Author/organization
- Title
- Publisher/journal/website
- Publication date
- URL if available

Format in markdown starting with:
## References

Group references by category.
"""
    
    sections["references"] = await generate_section(
        client, "References", base_system_prompt, references_prompt, search_results=formatted_search_results
    )
    log_success(f"Completed section {current_section}/{total_sections}: References")
    
    # We've already done the web searches at the beginning
    # No need to repeat them here
    
    # Generate Portfolio JSON
    log_info("Generating structured portfolio JSON data...")
    portfolio_data = await generate_portfolio_json(
        client, 
        [], 
        current_date,
        search_client=search_client,
        search_results=formatted_search_results if formatted_search_results else None
    )
    
    # Log a reminder about the position limits
    log_info("Validating portfolio positions count...")
    if "data" in portfolio_data and "assets" in portfolio_data["data"]:
        assets_count = len(portfolio_data["data"]["assets"])
        if assets_count < 10:
            log_warning(f"Portfolio contains only {assets_count} positions, fewer than the 10-15 required.")
        elif assets_count > 15:
            log_warning(f"Portfolio contains {assets_count} positions, exceeding the 10-15 limit specified in the prompt.")
        else:
            log_success(f"Portfolio contains {assets_count} positions, within the 10-15 range as specified.")
    
    # Add web search info as a message if available to the JSON generation
    if formatted_search_results and len(formatted_search_results) > 0:
        print("Adding web search data to JSON generation...")
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Combine all sections into full report
    section_order = [
        "executive_summary",
        "global_economy",
        "energy_markets",
        "commodities",
        "shipping",
        "portfolio_items",
        "benchmarking",
        "risk_assessment",
        "conclusion",
        "references"
    ]
    
    full_report = []
    for section_key in section_order:
        content = sections.get(section_key, "")
        if section_key == "executive_summary" and not content.startswith("# Orasis"):
            content = f"# Orasis Capital Multi-Asset Portfolio – {current_date}\n\n{content}"
        full_report.append(content)
    
    # Add the JSON at the end as a code block
    full_report.append("\n\n```json\n" + json.dumps(portfolio_data, indent=2) + "\n```")
    
    report_content = "\n\n".join(full_report)
    
    # Save the report content
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, "comprehensive_portfolio_report.md")
    with open(report_file, "w") as f:
        f.write(report_content)
    
    # Save portfolio data
    portfolio_file = os.path.join(output_dir, "comprehensive_portfolio_data.json")
    
    # Check if portfolio_data is a string and parse it if needed
    if isinstance(portfolio_data, str):
        try:
            log_info("Converting portfolio data from string to JSON object")
            portfolio_data = json.loads(portfolio_data)
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse portfolio data as JSON: {e}")
            # Try additional parsing methods if standard parsing fails
            if '```json' in portfolio_data:
                json_content = portfolio_data.split('```json', 1)[1].split('```')[0].strip()
                try:
                    portfolio_data = json.loads(json_content)
                    log_success("Successfully parsed JSON from markdown code block")
                except json.JSONDecodeError:
                    log_error("Failed to parse JSON from markdown code block")
    
    with open(portfolio_file, "w") as f:
        json.dump(portfolio_data, f, indent=2)
    
    print(f"Report generated successfully in {runtime:.2f} seconds")
    print(f"Report saved to: {report_file}")
    print(f"Portfolio data saved to: {portfolio_file}")
    
    # Display asset allocation summary
    if portfolio_data.get("status") == "success" and "data" in portfolio_data:
        assets = portfolio_data["data"].get("assets", [])
        print(f"\nPortfolio contains {len(assets)} assets:")
        
        for asset in assets[:5]:  # Show first 5 assets
            print(f"  {asset['asset_name']}: {asset['weight']}% - {asset['recommendation']}")
        
        if len(assets) > 5:
            print(f"  ... and {len(assets) - 5} more assets")
        
        # Show category summary
        if "summary" in portfolio_data["data"] and "by_category" in portfolio_data["data"]["summary"]:
            print("\nAllocation by category:")
            for category, weight in portfolio_data["data"]["summary"]["by_category"].items():
                print(f"  {category}: {weight}%")
        
        # Show references count
        references = portfolio_data["data"].get("references", [])
        if references:
            print(f"\nReport includes {len(references)} source references")
            # Show sample of reference categories
            categories = {}
            for ref in references:
                cat = ref.get("category", "Uncategorized")
                categories[cat] = categories.get(cat, 0) + 1
            
            print("Reference categories:")
            for cat, count in categories.items():
                print(f"  {cat}: {count} sources")
    
    return {
        "report": report_content,
        "portfolio_data": portfolio_data,
        "runtime": runtime
    }

if __name__ == "__main__":
    asyncio.run(generate_investment_portfolio())
    