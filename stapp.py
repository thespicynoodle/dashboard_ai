#!/usr/bin/env python3
"""
Streamlit app that combines the dashboard planning and HTML generation process into a single workflow.
It takes a CSV file and user instructions, generates a structured dashboard plan,
and then uses that plan to create a complete, interactive HTML dashboard.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
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
- **Top Row**: 3-4 cards for key metrics.
- **Bottom Section**: 3-5 charts for visual insights. Can be in two rows if necessary 

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

    def analyze_csv_data(self, df: pd.DataFrame) -> Tuple[str, Dict[str, Dict]]:
        """Analyzes a CSV DataFrame and returns a preview and column analysis."""
        try:
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
            raise RuntimeError(f"Error analyzing CSV data: {e}")

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

**CRITICAL CHART SIZING REQUIREMENTS:**
- Each chart container MUST have a fixed height of exactly 400px using CSS (height: 400px !important;)
- Charts MUST maintain their aspect ratio and not stretch vertically
- Use Chart.js responsive: true and maintainAspectRatio: false options
- Chart containers should be wrapped in divs with proper Bootstrap column classes (col-md-4)
- Add CSS rule: .chart-container {{ height: 400px !important; position: relative; }}
- Ensure charts fill their containers properly without overflow

Generate the complete HTML code now. Make a very snazzy dashboard that is production ready and has no errors and no weird stretching charts.

ONLY OUTPUT THE HTML 
"""

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
        return html_content
    except Exception as e:
        raise RuntimeError(f"API call for HTML generation failed: {e}")

# --- Streamlit App ---

def main():
    """Main Streamlit app function."""
    st.set_page_config(
        page_title="Dashboard Generator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 Dashboard Generator")
    st.markdown("---")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📁 Upload & Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file to generate a dashboard"
        )
        
        # Instructions input
        instructions = st.text_area(
            "💡 Dashboard Instructions",
            placeholder="e.g., 'Focus on sales by region', 'Show customer demographics'",
            help="Describe what you want to focus on in your dashboard"
        )
        
        # Output filename
        output_filename = st.text_input(
            "📄 Output Filename",
            value="dashboard.html",
            help="Name for the generated HTML file"
        )
    
    # Main content area
    if uploaded_file is not None and instructions.strip():
        try:
            # Read the uploaded CSV
            df = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("📈 Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", f"{len(df.columns):,}")
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            # Generate button
            if st.button("🚀 Generate Dashboard", type="primary", use_container_width=True):
                with st.spinner("Generating your dashboard..."):
                    
                    # Step 1: Plan the Dashboard
                    st.info("🤖 Step 1: Planning the dashboard...")
                    
                    planner = DashboardPlanner()
                    csv_preview, column_analysis = planner.analyze_csv_data(df)
                    dashboard_plan = planner.generate_plan(csv_preview, column_analysis, instructions)
                    
                    st.success("✅ Dashboard plan generated!")
                    
                    # Show the plan
                    with st.expander("📋 View Dashboard Plan"):
                        st.json(dashboard_plan.model_dump())
                    
                    # Step 2: Generate HTML
                    st.info("🤖 Step 2: Generating HTML dashboard...")
                    
                    html_content = generate_dashboard_html(dashboard_plan, uploaded_file.name)
                    
                    st.success("✅ HTML dashboard generated!")
                    
                    # Display results
                    st.subheader("🎉 Dashboard Generated Successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plan download
                        plan_json = dashboard_plan.model_dump_json(indent=2)
                        plan_filename = f"{Path(output_filename).stem}_plan.json"
                        st.download_button(
                            label="📋 Download Plan (JSON)",
                            data=plan_json,
                            file_name=plan_filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col2:
                        # HTML download
                        st.download_button(
                            label="🌐 Download Dashboard (HTML)",
                            data=html_content,
                            file_name=output_filename,
                            mime="text/html",
                            use_container_width=True
                        )
                    
                    # Instructions
                    st.info(f"💡 Download the HTML file and open it in your browser, then upload '{uploaded_file.name}' to see your dashboard in action!")
                    
                    # Show HTML preview (optional)
                    with st.expander("👀 Preview Generated HTML Code"):
                        st.code(html_content[:2000] + "..." if len(html_content) > 2000 else html_content, language="html")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    elif uploaded_file is None:
        st.info("👆 Please upload a CSV file to get started")
    
    elif not instructions.strip():
        st.info("💡 Please provide instructions for your dashboard")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and OpenAI")

if __name__ == "__main__":
    main()