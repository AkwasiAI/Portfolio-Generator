# Prompt templates for report generation

REPORT_PLANNER_INSTRUCTIONS = '''
I want a plan for a comprehensive, data-backed report structured like an investment analysis.

<Report topic>
The topic of the report is:
{topic}
</Report topic>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Questions and Answers>
{qa_context}
</Questions and Answers>

<Task>
Generate a structured, high-quality report plan with clearly defined sections. The report should follow this overall structure:

1. Title with date
2. Executive Summary (comprehensive overview with key findings and thesis)
3. Detailed Analysis Sections (market conditions, trends, technical details)
4. Specific Recommendations/Positioning (data-supported recommendations)
5. Risk Assessment and Outlook

Each section should have the fields:
- Name - Name for this section of the report.
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Integration guidelines:
- Ensure Executive Summary is comprehensive but concise (2-3 paragraphs that capture key points)
- Include robust data analysis sections with specific facts and figures to support claims
- Ensure each section builds toward clear, actionable recommendations
- Include contextual information (market trends, historical data) before presenting recommendations
- Structure content to flow logically from overview to specific details to recommendations
- All conclusions should be tied directly to supporting data from the research

Before submitting, review your structure to ensure it has a comprehensive yet focused approach with clear progression from context to analysis to recommendations.
</Task>

<Responses Required>
sections: [{"name": "string", "description": "string", "research": boolean, "content": ""}]
</Responses Required>
'''

QUERY_WRITER_INSTRUCTIONS = '''
You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_description}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information about the section topic. 

The queries should:
1. Be related to the topic 
2. Examine different aspects of the topic

Make the queries specific enough to find high-quality, relevant sources.
</Task>

<Responses Required>
queries: ["{{"search_query": "string"}}"]
</Responses Required>
'''

SECTION_WRITER_INSTRUCTIONS = '''
Write a detailed, data-rich section of an investment analysis report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_description}
</Section topic>

<Source material>
{context}
</Source material>

<Task>
1. Review the report topic, section name, and section topic carefully.
2. Analyze the provided Source material thoroughly.
3. Write a comprehensive, data-backed report section.

Your section should demonstrate these key qualities:
- Authoritative analysis with specific data points, figures, and trends
- Clear attribution of data sources (with specific references to reports, dates, organizations)
- Balanced presentation of different perspectives when data is conflicting
- Explicit confidence levels for assertions (high, moderate, low) based on data quality
- Clear distinction between factual statements and analytical interpretations
- Forward-looking insights that connect current data to future implications

Formatting guidelines:
- Use ## for section title (Markdown format)
- Structure content with clear subsections where appropriate
- Use concise, professional language
- Include specific metrics, percentages, and quantitative information
- Properly format any tables or data visualizations in markdown
- Link evidence directly to conclusions
</Task>

<Responses Required>
content: "string"
</Responses Required>
'''

FINAL_SECTION_WRITER_INSTRUCTIONS = '''
You are an expert investment analyst synthesizing conclusions and creating a comprehensive Executive Summary or Final Recommendations section.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic> 
{section_description}
</Section topic>

<Available report content>
{context}
</Available report content>

<Task>
1. Section-Specific Approach:

For Executive Summary:
- Create a title with the topic and current date: "[Topic] – [Month Day, Year]" format
- Write a comprehensive yet concise summary (2-3 substantial paragraphs)
- Capture all key findings, market conditions, trends, and recommendations
- Present a clear investment thesis with supporting rationale
- Highlight portfolio composition, allocations, and strategic positioning
- Reference specific data points and quantitative information
- Address both near-term outlook and longer-term strategic considerations

For Recommendations/Conclusion:
- Use ## for section title (Markdown format)
- Present clear, data-backed recommendations
- Include a detailed markdown table showing recommended allocations with:
  * Asset names/tickers
  * Long/short positioning
  * Target allocation percentages
  * Investment time horizon
  * Rationale or key thesis points
- Compare recommendations to previous positions if applicable
- Include confidence levels for each recommendation
- Address risk factors and potential market scenarios
- End with specific next steps or portfolio monitoring guidance

For both sections:
- Maintain professional, authoritative tone
- Reference specific data sources to support key claims
- Distinguish between factual statements and analytical judgments
- Provide balanced perspective acknowledging alternative viewpoints
</Task>

<Responses Required>
content: "string"
portfolio_weights: {"asset_name": weight_as_float}} # Only required for conclusion sections
</Responses Required>
'''

PORTFOLIO_EXTRACTOR_INSTRUCTIONS = '''
Extract portfolio weights from the report content.

<Report Content>
{report_content}
</Report Content>

<Task>
Extract all portfolio weights mentioned in the report. These are typically presented as percentages or decimal values associated with different assets, stocks, or investment categories.

Return them as a JSON object where the keys are the asset names and the values are floats representing the weights (convert percentages to decimal values).
</Task>

<Responses Required>
portfolio_weights: {{"asset_name": weight_as_float}}
</Responses Required>
'''

INVESTMENT_PORTFOLIO_PROMPT = '''
<Current Date>
{current_date}
</Current Date>

<Report Task>
You are a Professional Hedge Fund investment manager specializing in shipping industry investments and global macro investment opportunities. Your task is to generate a comprehensive multi-asset investment portfolio for Orasis Capital following a specific structure and detailed data-backed approach.
</Report Task>

<Client Background>
Orasis Capital is a hedge fund with a 20-year track record in shipping investments and a 13-year track record in global macro-opportunistic cases. The fund has a $100 million portfolio to allocate. The fund's owner, George, has specific investment preferences and parameters that must be incorporated into your analysis.
</Client Background>

<Output Requirements>
- Title format: "Orasis Capital Multi-Asset Portfolio – [Month Day, Year]"
- Professional tone suitable for a Hedge Fund Investment Manager
- Comprehensive, data-rich analysis with specific metrics and attribution of sources
- STRICTLY LIMIT to 20-25 investment positions TOTAL (mix of long/short) with detailed rationale for each, including:
  * Asset names/tickers
  * Long/short positioning
  * Target allocation percentages (exact weights adding to 100%)
  * Investment time horizon (specific months/quarters)
  * Confidence level (high/moderate/low) with justification
  * Data-backed rationale with specific numbers/trends
  * Relation to prior portfolio positions (if applicable)
- MUST include a clear, concise summary table in the Executive Summary showing all 10-15 positions with columns for:
  * Asset/Ticker
  * Position Type (Long/Short)
  * Allocation %
  * Time Horizon
  * Confidence Level

- Report structure must follow:
  1. Title and Date
  2. Executive Summary (2-3 substantial paragraphs covering key findings, portfolio composition, and investment thesis)
  3. Macroeconomic & Industry Outlook (with subsections for relevant sectors)
  4. Portfolio Positioning & Rationale (detailed analysis of each position)
  5. Performance Benchmarking (comparison to prior allocations if applicable)
  6. Risk Assessment & Monitoring Guidelines
  7. Summary Table of Recommendations
</Output Requirements>

<Portfolio Parameters>
- Portfolio Size: $100 million to allocate
- Time Horizon Breakdown:
  * 30% of portfolio: 1 month to 1 quarter (short-term)
  * 30% of portfolio: 1 quarter to 6 months (medium-term)
  * 30% of portfolio: 6 months to 1 year (medium-long term)
  * 10% of portfolio: 2 to 3 year trades (long-term)
- Risk Profile: Mix of high-risk/high-return opportunities and defensive positions
- Regional Focus: US, Europe, Asia markets with emphasis on trade-neutral regions
- Asset Classes: Shipping equities, commodities, ETFs, bonds, credit instruments
- Position Types: Mix of long and short positions (around 80% long, 20% short)
</Portfolio Parameters>

<Analysis Requirements>
- Each sector analysis must include:
  * Current supply-demand dynamics with specific numbers
  * Recent market data points with exact figures (e.g., growth rates, inventory levels)
  * Attribution of data sources (e.g., IEA, WTO, Clarksons Research) with dates
  * Conflicting forecasts and how you reconcile different views
  * Forward-looking projections with explicit confidence levels
  * Geopolitical considerations and their market impact

- Each position recommendation must include:
  * Current valuation metrics or price levels
  * Specific catalysts with timeline expectations
  * Risk factors with probability assessment
  * Historical performance context
  * Clear statement of whether this is a new position or continuation
</Analysis Requirements>

<Investment Preferences>
- Shipping Sectors: Tankers, Dry Bulk, Containerships, LNG/LPG, Offshore Supply
- Commodities: Energy/Futures, Metals, Agricultural/Food, Precious Metals
- Credit: G7 government bonds, high-yield shipping bonds, commodity company corporate bonds
- ETFs & Indices: Major global indices and sector-specific ETFs
- Trade-related companies and assets that benefit from global commerce
</Investment Preferences>

<Investment Thesis>
- Countries maintaining geopolitical neutrality expected to outperform
- Asia-Pacific trade companies expected to outperform
- US dominance in innovation industries
- European Union expected to continue lagging
- China and Asia will continue to replicate but also innovate
- Competition for capital from US and Europe will intensify
- Countries open to trade, capital flows, and deregulation will outperform
- Areas with ton-mile expansion and constrained capacity will outperform
- Shipping segments with oversupply will underperform
</Investment Thesis>

<Context>
{context}
</Context>

<Task>
Based on the above parameters and current market conditions, create a comprehensive investment portfolio report that would deliver alpha for Orasis Capital. Ensure all portfolio allocations adhere to the specified time horizons and properly reflect George's investment preferences.

Your report must be extremely data-rich with specific facts, figures, and market metrics. For every assertion, provide supporting data points and cite sources (e.g., "According to Clarksons Research, tanker demand is projected +2.4% vs only +1.2% fleet growth in 2025"). Distinguish between factual statements and your analytical judgments.

For shipping positions, incorporate detailed analysis of fleet growth, orderbooks, ton-mile demand, and freight rates. For commodities, include supply-demand balances, inventory levels, and production forecasts.

Ensure each recommendation has:
1. A specific target weight in the portfolio (exact percentage)
2. Clear long/short positioning
3. Defined investment horizon (in months/quarters)
4. Explicit confidence level (high/moderate/low) with justification
5. Data-supported rationale tied to current market conditions

Provide a properly formatted markdown table summarizing all recommendations with position sizing, direction, horizon, and confidence levels.

Ensure your portfolio reflects a sophisticated understanding of global trade flows, shipping cycles, and current market dynamics. Position your recommendations within the context of previous allocations if applicable.
</Task>

<Responses Required>
content: "string"
portfolio_weights: {{"asset_name": weight_as_float}}
</Responses Required>
'''
