#!/usr/bin/env python3
"""
This script combines the dashboard planning and HTML generation process into a single workflow.
It takes a CSV file and user instructions, generates a structured dashboard plan,
and then uses that plan to create a complete, interactive HTML dashboard.
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Pydantic Models & Enums ---

class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    AREA = "area"

class CardType(str, Enum):
    METRIC = "metric"
    KPI = "kpi"
    COUNT = "count"
    PERCENTAGE = "percentage"

class AggregationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    MAX = "max"
    MIN = "min"
    UNIQUE = "unique"

class DashboardCard(BaseModel):
    """Represents a single dashboard card/metric"""
    title: str = Field(..., description="Clear, concise title for the card")
    value_expression: str = Field(..., description="How to calculate the value (e.g., 'df[\"sales\"].sum()')")
    card_type: CardType = Field(..., description="Type of card being displayed")
    description: str = Field(..., description="Brief explanation of what this metric represents")
    format_hint: str = Field(..., description="How to format the value (e.g., '${:,.2f}', '{:,}')")
    columns_used: List[str] = Field(..., description="List of column names used in this calculation")

class DashboardChart(BaseModel):
    """Represents a single dashboard chart"""
    title: str = Field(..., description="Clear, descriptive title for the chart")
    chart_type: ChartType = Field(..., description="Type of chart to create")
    x_axis: str = Field(..., description="Column name for x-axis")
    y_axis: str = Field(..., description="Column name for y-axis")
    aggregation: Optional[AggregationType] = Field(None, description="Aggregation method if needed")
    group_by: Optional[str] = Field(None, description="Column to group by for aggregations")
    description: str = Field(..., description="What insights this chart is meant to reveal")
    columns_used: List[str] = Field(..., description="List of column names used in this chart")
    categories: Optional[List[str]] = Field(None, description="List of all unique values for a categorical column to ensure all are represented.")

class DashboardPlan(BaseModel):
    """Complete dashboard plan with cards and charts"""
    title: str = Field(..., description="Overall dashboard title")
    description: str = Field(..., description="Brief description of the dashboard's purpose")
    cards: List[DashboardCard] = Field(..., min_items=3, max_items=3, description="Exactly 3 dashboard cards")
    charts: List[DashboardChart] = Field(..., min_items=3, max_items=3, description="Exactly 3 dashboard charts")

# --- Dashboard Planner Class ---

class DashboardPlanner:
    """
    Analyzes CSV data and generates structured dashboard plans using an AI model.
    """
    PLANNING_PROMPT_TEMPLATE = """
You are an expert data analyst. Your task is to analyze CSV data and create a comprehensive dashboard plan based on the data and user requirements.

## INPUT DATA ANALYSIS
- **CSV Preview**: First 5 rows of the dataset.
- **Column Analysis**: Data type, unique values, missing values, and sample values for each column.
- **User Instructions**: Specific focus areas for the dashboard.

## DASHBOARD LAYOUT REQUIREMENTS
- **Top Row**: 3 cards for key metrics.
- **Bottom Section**: 3 charts for visual insights.

## CATEGORICAL DATA HANDLING
- For charts using a categorical column (e.g., for an axis or grouping), you MUST identify all its unique values from the 'sample_values' in the column analysis.
- Populate the 'categories' field in the chart plan with this list of unique values. This ensures the final dashboard represents all categories, even if some have no data in a particular view.

## YOUR TASK
Analyze the following data and create a dashboard plan.
"""

    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4.1"):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def analyze_csv_file(self, csv_path: str) -> Tuple[str, Dict[str, Dict]]:
        """Analyzes a CSV file and returns a preview and column analysis."""
        try:
            df = pd.read_csv(csv_path)
            csv_preview = df.head().to_csv(index=False)
            column_analysis = {}
            for col in df.columns:
                col_info = {
                    "data_type": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                    "missing_count": int(df[col].isnull().sum()),
                }
                # For categorical columns (object type with low cardinality), get all unique values
                if df[col].dtype == 'object' and df[col].nunique() > 0 and df[col].nunique() < 50: # Heuristic for categorical
                    col_info["sample_values"] = df[col].dropna().unique().tolist()
                elif df[col].dtype == 'object': # For other object columns, just show a sample
                     col_info["sample_values"] = df[col].dropna().unique()[:5].tolist()
                column_analysis[col] = col_info
            return csv_preview, column_analysis
        except Exception as e:
            raise RuntimeError(f"Error analyzing CSV file: {e}")

    def generate_plan(self, csv_preview: str, column_analysis: Dict[str, Dict], user_instructions: str) -> DashboardPlan:
        """Generates a dashboard plan using the OpenAI API with structured output."""
        column_analysis_str = json.dumps(column_analysis, indent=2)
        
        prompt = (
            f"{self.PLANNING_PROMPT_TEMPLATE}\n\n"
            f"### CSV PREVIEW:\n```csv\n{csv_preview}\n```\n\n"
            f"### COLUMN ANALYSIS:\n```json\n{column_analysis_str}\n```\n\n"
            f"### USER INSTRUCTIONS:\n{user_instructions}\n\n"
            "Now, generate the dashboard plan."
        )

        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert that generates dashboard plans."},
                    {"role": "user", "content": prompt}
                ],
                response_format=DashboardPlan,
                temperature=0
            )
            
            dashboard_plan = completion.choices[0].message.parsed
            if completion.choices[0].message.refusal:
                raise RuntimeError(f"Dashboard plan generation refused: {completion.choices[0].message.refusal}")

            return dashboard_plan
        except Exception as e:
            raise RuntimeError(f"Failed to generate dashboard plan: {e}")

# --- HTML Generator Functions ---

def generate_dashboard_html(dashboard_plan: DashboardPlan, csv_filename: str, model: str = "gpt-4.1") -> str:
    """Generates dashboard HTML from a DashboardPlan object using an AI model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create a detailed prompt for the HTML generation
    card_details = "\n".join(
        f"- Card {i+1}: '{card.title}' to show '{card.description}'. Calculation: `{card.value_expression}`."
        for i, card in enumerate(dashboard_plan.cards)
    )
    
    chart_details_list = []
    for i, chart in enumerate(dashboard_plan.charts):
        detail = f"- Chart {i+1}: A '{chart.chart_type.value}' chart titled '{chart.title}' with X-axis '{chart.x_axis}' and Y-axis '{chart.y_axis}'."
        if chart.categories:
            categories_str = ", ".join(f"'{c}'" for c in chart.categories)
            detail += f" For the '{chart.x_axis}', ensure all of the following categories are represented in order: [{categories_str}]."
        chart_details_list.append(detail)
    chart_details = "\n".join(chart_details_list)


    prompt = f"""
Create a complete, single-file HTML dashboard.

**Dashboard Title:** {dashboard_plan.title}
**Description:** {dashboard_plan.description}

**Layout:**
- A top row with 3 metric cards.
- A bottom row with 3 charts side-by-side.

**Component Details:**
{card_details}
{chart_details}

**Technical Requirements:**
- Use Bootstrap 5 for styling (via CDN).
- Use Chart.js for charts (via CDN).
- Use Papa Parse for in-browser CSV parsing (via CDN).
- Include a file upload input to load the CSV data ('{csv_filename}').
- The dashboard should be responsive, modern, and professional.
- All CSS and JavaScript must be embedded within the HTML file.
- Implement the card calculations and chart rendering using JavaScript based on the uploaded CSV data.
- For charts with specified categories, the JavaScript code MUST pre-populate the chart labels with all the provided categories to ensure they are all displayed, even if they have a value of zero in the current dataset.
- Ensure robust error handling for file parsing and chart creation.
- Use very professional whites and navy blues when building the dashboard

Generate the complete HTML code now. Make a very snazzy dashboard that is production ready and has no errors and wierd charts.

ONLY OUTPUT THE HTML 
"""

    print("🤖 Step 3: Generating HTML dashboard via AI...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert web developer specializing in creating data dashboards."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        html_content = response.choices[0].message.content
        # Clean up potential markdown formatting
        if html_content.strip().startswith("```html"):
            html_content = html_content.strip()[7:-4]
        print("✅ HTML generated successfully!")
        return html_content
    except Exception as e:
        raise RuntimeError(f"API call for HTML generation failed: {e}")

def save_to_file(content: str, filename: str):
    """Saves content to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 File saved to: {filename}")

# --- Main Execution ---

def main():
    """Main function to run the combined planner and generator."""
    print("🚀 Dashboard Generator")
    print("=" * 50)
    
    # Get CSV file path
    while True:
        csv_file = input("Enter the path to your CSV file: ").strip()
        if os.path.exists(csv_file):
            break
        else:
            print(f"❌ Error: CSV file not found at '{csv_file}'")
            print("Please enter a valid file path.")
    
    # Get user instructions
    print("\n💡 Enter your dashboard instructions (e.g., 'Focus on sales by region', 'Show customer demographics'):")
    instructions = input("Instructions: ").strip()
    
    # Get output filename (optional)
    output_file = input("\n📁 Enter output HTML filename (or press Enter for 'dashboard.html'): ").strip()
    if not output_file:
        output_file = 'dashboard.html'

    try:
        # --- Step 1: Plan the Dashboard ---
        print(f"\n🚀 Starting dashboard generation for '{csv_file}'...")
        print("🤖 Step 1: Planning the dashboard...")
        planner = DashboardPlanner()

        print("Analyzing CSV data...")
        csv_preview, column_analysis = planner.analyze_csv_file(csv_file)
        
        print("Generating dashboard plan via AI...")
        dashboard_plan = planner.generate_plan(csv_preview, column_analysis, instructions)
        
        plan_filename = f"{Path(output_file).stem}_plan.json"
        save_to_file(dashboard_plan.model_dump_json(indent=2), plan_filename)
        print("✅ Dashboard plan generated and saved.")

        # --- Step 2: Generate the HTML ---
        print("\n🤖 Step 2: Proceeding to HTML generation...")
        csv_basename = os.path.basename(csv_file)
        html_content = generate_dashboard_html(dashboard_plan, csv_basename)
        
        # --- Step 3: Save the final output ---
        save_to_file(html_content, output_file)

        print("\n🎉 --- SUCCESS! --- 🎉")
        print(f"📊 Plan: {plan_filename}")
        print(f"🌐 Dashboard: {output_file}")
        print(f"💡 Open '{output_file}' in your browser and upload '{csv_basename}' to see your dashboard.")

    except (ValueError, RuntimeError, Exception) as e:
        print(f"\n❌ --- AN ERROR OCCURRED --- ❌")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()