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
from src.portfolio_generator.web_search import PerplexitySearch, format_search_results

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

def save_prompts_to_file(current_date, base_system_prompt, exec_summary_prompt, global_economy_prompt,
                      energy_markets_prompt, commodities_prompt, shipping_prompt, asset_prompt,
                      portfolio_prompt, conclusion_prompt, references_prompt, search_queries):
    """Save all prompts used in the report generation to a text file in the output folder."""
    try:
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Create filename with current date
        prompts_file_path = f"output/investment_portfolio_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(prompts_file_path, "w") as f:
            f.write(f"# Investment Portfolio Prompts - Generated on {current_date}\n\n")
            
            # Base system prompt
            f.write("## Base System Prompt\n")
            f.write(base_system_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Executive Summary prompt
            f.write("## Executive Summary Prompt\n")
            f.write(exec_summary_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Global Economy prompt
            f.write("## Global Economy Prompt\n")
            f.write(global_economy_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Energy Markets prompt
            f.write("## Energy Markets Prompt\n")
            f.write(energy_markets_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Commodities prompt
            f.write("## Commodities Prompt\n")
            f.write(commodities_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Shipping prompt
            f.write("## Shipping Sectors Prompt\n")
            f.write(shipping_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Asset List prompt
            f.write("## Asset List Generation Prompt\n")
            f.write(asset_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Portfolio Positions prompt
            f.write("## Portfolio Positions Prompt\n")
            f.write(portfolio_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Conclusion prompt
            f.write("## Conclusion and Summary Prompt\n")
            f.write(conclusion_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # References prompt
            f.write("## References Prompt\n")
            f.write(references_prompt)
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Web Search Queries
            f.write("## Web Search Queries\n")
            f.write("The following search queries were used to gather market data:\n\n")
            for i, query in enumerate(search_queries, 1):
                f.write(f"{i}. {query}\n")
        
        log_success(f"Saved all prompts to {prompts_file_path}")
    except Exception as e:
        log_error(f"Error saving prompts to file: {e}")

async def extract_portfolio_data_from_sections(sections, current_date):
    """Extract portfolio data from the generated report sections to create a structured JSON."""
    # Create the base JSON structure that matches the expected format
    portfolio_json = {
        "status": "success",
        "data": {
            "report_date": current_date,
            "assets": [],
            "summary": {
                "by_category": {},
                "by_region": {},
                "by_recommendation": {}
            }
        }
    }
    
    try:
        # Extract data from the executive summary section which has the summary table
        exec_summary = sections.get("executive_summary", "")
        portfolio_items = sections.get("portfolio_items", "")
        all_sections_text = "".join(sections.values())
        
        # Use regex to extract the portfolio table from executive summary
        import re
        
        # Extract assets from markdown table in the executive summary
        table_pattern = r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|"
        assets = []
        category_allocations = {}
        region_allocations = {}
        recommendation_allocations = {}
        
        # First pass: gather all assets from the executive summary table
        matches = re.findall(table_pattern, exec_summary)
        for match in matches:
            # Skip header rows or non-asset rows
            if any(header in match[0].lower() for header in ["asset", "ticker", "---"]) or not match[0].strip():
                continue
                
            # Process asset data
            asset_name = match[0].strip()
            position_type = match[1].strip()
            allocation = match[2].strip().replace("%", "").strip()
            time_horizon = match[3].strip()
            confidence = match[4].strip()
            
            # Extract asset details from portfolio section
            asset_info = {}
            
            # Look for detailed information about this asset in the entire report
            asset_sections = re.findall(rf"{re.escape(asset_name)}[\s\S]*?(?=\n\n\d+\.|$)", all_sections_text)
            asset_text = "\n".join(asset_sections) if asset_sections else ""
            
            # Extract category
            category = "Equity"  # Default category
            category_match = re.search(r"[Cc]ategory[:\s]+([^\n.,;]+)", asset_text)
            if category_match:
                category = category_match.group(1).strip()
            
            # Extract region
            region = "Global"  # Default region
            region_match = re.search(r"[Rr]egion[:\s]+([^\n.,;]+)", asset_text)
            if region_match:
                region = region_match.group(1).strip()
            
            # Extract rationale
            rationale = ""
            rationale_match = re.search(r"[Rr]ationale[:\s]+([^\n.]+)", asset_text)
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            else:
                # If no specific rationale, try to find any sentence with the asset name
                rationale_sentences = re.findall(rf"[^.!?]*{re.escape(asset_name)}[^.!?]*[.!?]", all_sections_text)
                if rationale_sentences:
                    rationale = rationale_sentences[0].strip()
            
            # Determine recommendation based on position type
            recommendation = ""
            if position_type.lower() == "long":
                recommendation = "Buy Long"
            elif position_type.lower() == "short":
                recommendation = "Sell Short"
            
            # Standardize time horizon format
            horizon = time_horizon
            if "month" in time_horizon.lower() or "m" in time_horizon.lower():
                if any(term in time_horizon.lower() for term in ["1", "2", "3", "short"]):
                    horizon = "Short (1-3M)"
                elif any(term in time_horizon.lower() for term in ["3", "4", "5", "6", "medium"]):
                    horizon = "Medium (3-6M)"
                elif any(term in time_horizon.lower() for term in ["6", "7", "8", "9", "10", "11", "12", "long"]):
                    horizon = "Long (6-12M)"
            
            # Create formatted asset entry
            asset = {
                "asset_name": asset_name,
                "category": category,
                "region": region,
                "weight": int(allocation) if allocation.isdigit() else 0,
                "horizon": horizon,
                "recommendation": recommendation,
                "rationale": rationale
            }
            assets.append(asset)
            
            # Update category allocations
            if category not in category_allocations:
                category_allocations[category] = 0
            category_allocations[category] += int(allocation) if allocation.isdigit() else 0
            
            # Update region allocations
            if region not in region_allocations:
                region_allocations[region] = 0
            region_allocations[region] += int(allocation) if allocation.isdigit() else 0
            
            # Update recommendation allocations
            if recommendation not in recommendation_allocations:
                recommendation_allocations[recommendation] = 0
            recommendation_allocations[recommendation] += int(allocation) if allocation.isdigit() else 0
        
        # Set the assets in the JSON
        portfolio_json["data"]["assets"] = assets
        portfolio_json["data"]["summary"]["by_category"] = category_allocations
        portfolio_json["data"]["summary"]["by_region"] = region_allocations
        portfolio_json["data"]["summary"]["by_recommendation"] = recommendation_allocations
        
        return portfolio_json
    except Exception as e:
        log_error(f"Error extracting portfolio data from sections: {e}")
        return {
            "status": "error",
            "data": {
                "report_date": current_date,
                "assets": [],
                "error": str(e)
            }
        }
    
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
    
    # Use the current date instead of a fixed date
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Start time for tracking runtime
    start_time = time.time()
    
    # Perform web searches upfront to have the data available for all API calls
    formatted_search_results = ""
    if search_client:
        try:
            log_info("Performing web searches for market data upfront...")
            
            # List of search queries focusing on financial news sources and market data
            # Limited to 20 queries for efficiency
            current_month_year = datetime.now().strftime("%B %Y")
            search_queries = [
                # Global Economy & Finance (4 queries)
                f"Bloomberg financial market analysis global economy {current_month_year}",
                f"Financial Times GDP growth forecasts by region {current_month_year}",
                f"Wall Street Journal global investment outlook {current_month_year}",
                f"Reuters market intelligence financial trends {current_month_year}",
                
                # Shipping & Transportation Finance (4 queries)
                f"Bloomberg shipping stock analysis maritime industry {current_month_year}",
                f"Financial Times Baltic Dry Index forecast {current_month_year}",
                f"MarineLink tanker market analysis rates {current_month_year}",
                f"Bloomberg container shipping industry financials {current_month_year}",
                
                # Energy Markets (4 queries)
                f"Bloomberg energy commodities market analysis {current_month_year}",
                f"Reuters oil price forecast investment {current_month_year}",
                f"S&P Global natural gas market report {current_month_year}",
                f"Financial Times LNG market investment outlook {current_month_year}",
                
                # Commodities & Investment (4 queries)
                f"Bloomberg commodities market analysis metals {current_month_year}",
                f"Reuters agricultural commodities investment {current_month_year}",
                f"Barron's commodity ETF performance {current_month_year}",
                f"Wall Street Journal metals market investment {current_month_year}",
                
                # Financial Markets & Investment (4 queries)
                f"Bloomberg investment portfolio strategy {current_month_year}",
                f"Morningstar ETF analysis sector performance {current_month_year}",
                f"Financial Times interest rates investment impact {current_month_year}",
                f"Wall Street Journal currency market investment strategy {current_month_year}"
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

IMPORTANT CLIENT CONTEXT - GEORGE (HEDGE FUND OWNER):
George, the owner of Orasis Capital, has specified the following investment preferences:

1. Risk Tolerance: Both high-risk opportunities and balanced investments with a mix of defensive and growth-oriented positions.

2. Time Horizon Distribution:
   - 30% of portfolio: 1 month to 1 quarter (short-term)
   - 30% of portfolio: 1 quarter to 6 months (medium-term)
   - 30% of portfolio: 6 months to 1 year (medium-long term)
   - 10% of portfolio: 2 to 3 year trades (long-term)

3. Investment Strategy: Incorporate both leverage and hedging strategies, not purely cash-based.

4. Regional Focus: US, Europe, and Asia, with specific attention to global trade shifts affecting China, Asia, Middle East, and Africa.

5. Commodity Interests: Wide range including crude oil futures, natural gas, metals, agricultural commodities, and related companies.

6. Shipping Focus: Strong emphasis on various shipping segments including tanker, dry bulk, container, LNG, LPG, and offshore sectors.

7. Credit Exposure: Include G7 10-year government bonds, high-yield shipping bonds, and corporate bonds of commodities companies.

8. ETF & Indices: Include major global indices (Dow Jones, S&P 500, NASDAQ, European indices, Asian indices) and other tradeable ETFs.

INVESTMENT THESIS:
Orasis Capital's core strategy is to capitalize on global trade opportunities, with a 20-year track record in shipping-related investments. The fund identifies shifts in global trade relationships that impact countries and industries, analyzing whether these impacts are manageable. Key focuses include monitoring changes in trade policies from new governments, geopolitical developments, and structural shifts in global trade patterns.

The firm believes trade flows are changing, with China, Asia, the Middle East, and Africa gaining more investment and trade volume compared to traditional areas like the US and Europe. Their research approach uses shipping (90% of global trade volume) as a leading indicator for macro investments, allowing them to identify shifts before they become widely apparent.

IMPORTANT CONSTRAINTS:
1. The ENTIRE report must be NO MORE than 13,000 words total. Optimize your content accordingly.
2. You MUST include a comprehensive summary table in the Executive Summary section.
3. Ensure all assertions are backed by specific data points or sources.
4. Use current data from 2024-2025 where available."""
    
    # Initialize section tracking variables
    total_sections = 10  # Total number of sections in the report
    current_section = 1  # Initialize the current section counter

    # Dictionary to store all sections
    sections = {}
    
    # 1. Generate Executive Summary
    log_info("Generating executive summary section...")
    exec_summary_prompt = f"""Generate an executive summary for the investment portfolio report.

Include current date ({current_date}) and the title format specified previously.
Summarize the key findings, market outlook, and high-level portfolio strategy.
Keep it clear, concise, and data-driven with specific metrics.

CRITICAL REQUIREMENT: You MUST include a comprehensive summary table displaying ALL portfolio positions (strictly limited to 20-25 total positions).
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
    # First, generate a list of 20-25 diverse assets across asset classes
    asset_prompt = """Create a list of 20-25 diverse investment assets that would be suitable for a trade-focused multi-asset portfolio.
Include a well-balanced mix of:
- Shipping equities (tankers, dry bulk, containers, LNG carriers, port operators)
- Energy equities and ETFs (oil, natural gas, LNG, renewable)
- Commodity producers and ETFs (metals, agricultural, industrial)
- Bonds and credit instruments (corporate, sovereign, treasury)
- Agricultural assets and related companies
- Infrastructure assets related to global trade
- Logistics and supply chain companies
- Financial services related to trade finance
- Currency and forex instruments

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

STRICTLY LIMIT to exactly 20-25 investment positions TOTAL (mix of long/short) with detailed rationale for each.
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
Ensure comprehensive diversification across different market sectors, particularly focusing on finance-related assets.
Do not add any positions beyond the 20-25 shown in the Executive Summary table.
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
    
    # Extract portfolio data from the generated sections
    log_info("Extracting portfolio data from generated report sections...")
    portfolio_json = await extract_portfolio_data_from_sections(sections, current_date)
    
    # Log a reminder about the position limits
    log_info("Validating portfolio positions count...")
    # Check the extracted portfolio data
    if "data" in portfolio_json and "assets" in portfolio_json["data"]:
        assets_count = len(portfolio_json["data"]["assets"])
        if assets_count < 20:
            log_warning(f"Portfolio contains only {assets_count} positions, fewer than the 20-25 required.")
        elif assets_count > 25:
            log_warning(f"Portfolio contains {assets_count} positions, more than the 20-25 required.")
    else:
        log_error("Failed to extract portfolio data properly.")
        
    # Convert to JSON string for storage
    portfolio_data = json.dumps(portfolio_json, indent=2)
    
    # Save all prompts to a text file for reference
    save_prompts_to_file(current_date, base_system_prompt, exec_summary_prompt, global_economy_prompt,
                        energy_markets_prompt, commodities_prompt, shipping_prompt, asset_prompt,
                        portfolio_prompt, conclusion_prompt, references_prompt, search_queries)
    
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
    if isinstance(portfolio_json, dict) and portfolio_json.get("status") == "success" and "data" in portfolio_json:
        assets = portfolio_json["data"].get("assets", [])
        print(f"\nPortfolio contains {len(assets)} assets:")
        
        for asset in assets[:5]:  # Show first 5 assets
            print(f"  {asset.get('asset_name', 'Unknown')}: {asset.get('weight', '0')}% - {asset.get('recommendation', 'No recommendation')}")
        
        if len(assets) > 5:
            print(f"  ... and {len(assets) - 5} more assets")
        
        # Show allocation by category
        if "summary" in portfolio_json["data"] and "by_category" in portfolio_json["data"]["summary"]:
            categories = portfolio_json["data"]["summary"]["by_category"]
            print("\nAllocation by category:")
            for category, weight in categories.items():
                print(f"  {category}: {weight}%")
        
        # Show allocation by region
        if "summary" in portfolio_json["data"] and "by_region" in portfolio_json["data"]["summary"]:
            regions = portfolio_json["data"]["summary"]["by_region"]
            print("\nAllocation by region:")
            for region, weight in regions.items():
                print(f"  {region}: {weight}%")
        
        # Show allocation by recommendation
        if "summary" in portfolio_json["data"] and "by_recommendation" in portfolio_json["data"]["summary"]:
            recommendations = portfolio_json["data"]["summary"]["by_recommendation"]
            print("\nAllocation by recommendation:")
            for recommendation, weight in recommendations.items():
                print(f"  {recommendation}: {weight}%")
        
        # Count the number of unique categories
        category_count = {}
        for asset in assets:
            cat = asset.get("category", "Uncategorized")
            if cat not in category_count:
                category_count[cat] = 0
            category_count[cat] += 1
        
        print("\nPosition count by category:")
        for category, count in category_count.items():
            if category:
                print(f"  {category}: {count} positions")
    
    return {
        "report": report_content,
        "portfolio_data": portfolio_data,
        "runtime": runtime
    }

if __name__ == "__main__":
    asyncio.run(generate_investment_portfolio())