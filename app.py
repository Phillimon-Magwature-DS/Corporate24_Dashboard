import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Corporate 24 Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/styles.css")

# Load company logo with circular mask
def load_logo():
    try:
        logo = Image.open("assets/logo.png")
        # Resize logo
        logo = logo.resize((150, 150))
        
        # Create circular mask
        bigsize = (logo.size[0] * 3, logo.size[1] * 3)
        mask = Image.new('L', bigsize, 0)
        draw = ImageDraw.Draw(mask) 
        draw.ellipse((0, 0) + bigsize, fill=255)
        mask = mask.resize(logo.size)
        
        # Apply mask
        logo.putalpha(mask)
        
        # Convert to base64
        buffered = BytesIO()
        logo.save(buffered, format="PNG")
        logo_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return logo_base64
    except Exception as e:
        st.warning(f"Logo not found or error loading logo: {str(e)}")
        return None

# Convert image to base64
def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Data cleaning function
def clean_data(df):
    # Convert date columns to datetime
    date_cols = ['DATE OF BIRTH', 'VISIT_DATE']
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # Clean telephone numbers
    if 'TELEPHONE' in df.columns:
        df['TELEPHONE'] = df['TELEPHONE'].astype(str).str.replace('.0', '', regex=False)
        df['TELEPHONE'] = df['TELEPHONE'].str.replace(r'\.', '', regex=True)
        df['TELEPHONE'] = df['TELEPHONE'].str.replace(r'E\+11', '', regex=True)
    
    # Clean medical aid names
    if 'MEDICAL AID' in df.columns:
        df['MEDICAL AID'] = df['MEDICAL AID'].str.strip().str.upper()
    
    # Clean sex/gender
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].str.strip().str.title()
        df['SEX'] = df['SEX'].replace({'Female': 'F', 'Male': 'M', 'F': 'Female', 'M': 'Male'})
    
    # Clean facility names
    if 'FACILITY' in df.columns:
        df['FACILITY'] = df['FACILITY'].str.strip().str.upper()
    
    # Clean branch IDs
    if 'BranchID' in df.columns:
        df['BranchID'] = df['BranchID'].str.strip().str.upper()
    
    return df

# Calculate age from date of birth
def calculate_age(dob):
    today = datetime.now()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# Load data function
@st.cache_data(ttl=3600)
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        df = clean_data(df)
        
        # Calculate age if date of birth exists
        if 'DATE OF BIRTH' in df.columns and pd.api.types.is_datetime64_any_dtype(df['DATE OF BIRTH']):
            df['AGE'] = df['DATE OF BIRTH'].apply(lambda x: calculate_age(x) if not pd.isnull(x) else None)
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Get available datasets in data directory
def get_available_datasets():
    data_files = []
    if os.path.exists("data"):
        for file in os.listdir("data"):
            if file.endswith(('.csv', '.xlsx', '.xls')):
                data_files.append(os.path.join("data", file))
    return data_files

# Main app function
def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Visualization"
    
    # Load logo
    logo_base64 = load_logo()
    
    # Sidebar styling
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #1e3c72;
            }
            .sidebar-title {
                color: white !important;
                text-align: center;
                margin-bottom: 20px;
            }
            .sidebar-section {
                margin-bottom: 30px;
            }
            .sidebar-section-title {
                color: #ffffff;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .st-bb {
                background-color: transparent;
            }
            .st-at {
                background-color: #ffffff;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                border: none;
                padding: 0.5rem 1rem;
                font-weight: 500;
                width: 100%;
                margin: 5px 0;
                transition: all 0.3s;
            }
            .stButton button:hover {
                background-color: #45a049;
                transform: scale(1.02);
            }
            .stSelectbox, .stTextInput, .stDateInput {
                background-color: white;
                border-radius: 5px;
            }
            .stRadio div {
                color: white;
            }
            .stRadio div label {
                color: white !important;
            }
            .stRadio div[role="radiogroup"] {
                background-color: rgba(255,255,255,0.1);
                padding: 10px;
                border-radius: 5px;
            }
            .stRadio div[role="radiogroup"] label {
                margin-right: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        # Dashboard title
        st.markdown("<h1 class='sidebar-title'>DASHBOARD</h1>", unsafe_allow_html=True)
        
        # Logo and company name
        if logo_base64:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                    <div style="width: 150px; height: 150px; border-radius: 50%; background-color: white; display: flex; justify-content: center; align-items: center; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                        <img src="data:image/png;base64,{logo_base64}" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                </div>
                <h3 style='text-align: center; color: white; margin-bottom: 30px;'>Corporate 24 Healthcare</h3>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("<h3 style='text-align: center; color: white; margin-bottom: 30px;'>Corporate 24 Healthcare</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data upload section
        st.markdown("<div class='sidebar-section'><h3 class='sidebar-section-title'>Data Management</h3></div>", unsafe_allow_html=True)
        
        # Option to select from existing datasets
        available_datasets = get_available_datasets()
        if available_datasets:
            selected_dataset = st.selectbox("Select from available datasets", [""] + available_datasets)
            if selected_dataset and selected_dataset != "":
                if st.button("Load Selected Dataset"):
                    st.session_state.df = load_data(selected_dataset)
                    if st.session_state.df is not None:
                        st.session_state.data_loaded = True
                        st.success("Dataset loaded successfully!")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            # Save the uploaded file to data directory
            if not os.path.exists("data"):
                os.makedirs("data")
            
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.df = load_data(file_path)
            if st.session_state.df is not None:
                st.session_state.data_loaded = True
                st.success("Dataset uploaded and loaded successfully!")
        
        # Close dataset button
        if st.session_state.data_loaded:
            if st.button("Close Current Dataset", key="close_dataset"):
                st.session_state.df = None
                st.session_state.data_loaded = False
                st.rerun()
        
        st.markdown("---")
        
        # Navigation
        st.markdown("<div class='sidebar-section'><h3 class='sidebar-section-title'>Navigation</h3></div>", unsafe_allow_html=True)
        page_options = ["Visualization", "Data View", "About"]
        st.session_state.current_page = st.radio("Go to", page_options, index=page_options.index(st.session_state.current_page))
        
        # Footer link
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; margin-top: 20px;">
                <p style="color: white; font-size: 14px;">
                    <a href="https://corp24med.com/" target="_blank" style="color: #ffffff; text-decoration: none;">
                        For More Info, Visit our Website
                    </a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Main content area styling
    st.markdown(
        """
        <style>
            .main .block-container {
                background-color: #f8f9fa;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .stTabs [role="tablist"] {
                background-color: #e9ecef;
                border-radius: 8px;
                padding: 5px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1e3c72 !important;
                color: white !important;
                border-radius: 5px;
                font-weight: bold;
            }
            .stTabs [aria-selected="false"] {
                color: #495057;
            }
            .stMetric {
                background-color: white;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-left: 4px solid #1e3c72;
            }
            .stExpander {
                background-color: white;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            .stDataFrame {
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            h1, h2, h3, h4, h5, h6 {
                color: #1e3c72;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        # Visualization Page
        if st.session_state.current_page == "Visualization":
            st.title("📊 Data Visualization")
            
            # Filters
            with st.expander("🔍 Filters", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                # Date filter
                if 'VISIT_DATE' in df.columns and pd.api.types.is_datetime64_any_dtype(df['VISIT_DATE']):
                    min_date = df['VISIT_DATE'].min().date()
                    max_date = df['VISIT_DATE'].max().date()
                    
                    with col1:
                        date_range = st.date_input(
                            "📅 Visit Date Range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        df = df[(df['VISIT_DATE'].dt.date >= start_date) & (df['VISIT_DATE'].dt.date <= end_date)]
                
                # Gender filter
                if 'SEX' in df.columns:
                    with col2:
                        gender_options = ['All'] + list(df['SEX'].dropna().unique())
                        selected_gender = st.selectbox("👥 Gender", gender_options)
                        if selected_gender != 'All':
                            df = df[df['SEX'] == selected_gender]
                
                # Medical Aid filter
                if 'MEDICAL AID' in df.columns:
                    with col3:
                        medical_aid_options = ['All'] + list(df['MEDICAL AID'].dropna().unique())
                        selected_medical_aid = st.selectbox("🏥 Medical Aid", medical_aid_options)
                        if selected_medical_aid != 'All':
                            df = df[df['MEDICAL AID'] == selected_medical_aid]
                
                # Additional filters
                col4, col5, col6 = st.columns(3)
                
                # Facility filter
                if 'FACILITY' in df.columns:
                    with col4:
                        facility_options = ['All'] + list(df['FACILITY'].dropna().unique())
                        selected_facility = st.selectbox("🏢 Facility", facility_options)
                        if selected_facility != 'All':
                            df = df[df['FACILITY'] == selected_facility]
                
                # Branch filter
                if 'BranchID' in df.columns:
                    with col5:
                        branch_options = ['All'] + list(df['BranchID'].dropna().unique())
                        selected_branch = st.selectbox("🏷️ Branch", branch_options)
                        if selected_branch != 'All':
                            df = df[df['BranchID'] == selected_branch]
                
                # Age filter
                if 'AGE' in df.columns:
                    with col6:
                        min_age, max_age = st.slider(
                            "👶👴 Age Range",
                            min_value=int(df['AGE'].min()),
                            max_value=int(df['AGE'].max()),
                            value=(int(df['AGE'].min()), int(df['AGE'].max())))
                        df = df[(df['AGE'] >= min_age) & (df['AGE'] <= max_age)]
            
            # Visualization tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview Metrics", 
                "👥 Demographics", 
                "🏥 Facility Analysis", 
                "📅 Temporal Trends",
                "🔍 Detailed Analysis"
            ])
            
            with tab1:
                st.subheader("Key Metrics")
                
                if not df.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Patients", len(df['PATIENT ID'].unique()))
                    
                    with col2:
                        st.metric("Total Visits", len(df))
                    
                    with col3:
                        if 'AGE' in df.columns:
                            avg_age = df['AGE'].mean()
                            st.metric("Average Age", f"{avg_age:.1f} years")
                    
                    with col4:
                        if 'SEX' in df.columns:
                            gender_dist = df['SEX'].value_counts(normalize=True)
                            if 'Female' in gender_dist.index:
                                female_pct = gender_dist['Female'] * 100
                                st.metric("Female Patients", f"{female_pct:.1f}%")
                    
                    # Medical Aid Distribution
                    if 'MEDICAL AID' in df.columns:
                        st.subheader("Medical Aid Distribution")
                        
                        medical_aid_counts = df['MEDICAL AID'].value_counts().reset_index()
                        medical_aid_counts.columns = ['Medical Aid', 'Count']
                        
                        fig = px.pie(
                            medical_aid_counts, 
                            values='Count', 
                            names='Medical Aid',
                            hole=0.3,
                            title="Medical Aid Distribution",
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Facility Distribution
                    if 'FACILITY' in df.columns:
                        st.subheader("Facility Distribution")
                        
                        facility_counts = df['FACILITY'].value_counts().reset_index()
                        facility_counts.columns = ['Facility', 'Count']
                        
                        fig = px.bar(
                            facility_counts,
                            x='Facility',
                            y='Count',
                            title="Visits by Facility",
                            color='Count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Demographic Analysis")
                
                if not df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gender Distribution
                        if 'SEX' in df.columns:
                            gender_counts = df['SEX'].value_counts().reset_index()
                            gender_counts.columns = ['Gender', 'Count']
                            
                            fig = px.pie(
                                gender_counts,
                                values='Count',
                                names='Gender',
                                title="Gender Distribution",
                                hole=0.3,
                                color_discrete_sequence=['#1e3c72', '#2a5298']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Age Distribution
                        if 'AGE' in df.columns:
                            fig = px.histogram(
                                df,
                                x='AGE',
                                nbins=20,
                                title="Age Distribution",
                                labels={'AGE': 'Age'},
                                color_discrete_sequence=['#1e3c72']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Age by Gender
                    if 'AGE' in df.columns and 'SEX' in df.columns:
                        st.subheader("Age Distribution by Gender")
                        
                        fig = px.box(
                            df,
                            x='SEX',
                            y='AGE',
                            color='SEX',
                            title="Age Distribution by Gender",
                            color_discrete_map={'Female': '#2a5298', 'Male': '#1e3c72'}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top Surnames
                    if 'SURNAME' in df.columns:
                        st.subheader("Top 10 Surnames")
                        
                        top_surnames = df['SURNAME'].value_counts().head(10).reset_index()
                        top_surnames.columns = ['Surname', 'Count']
                        
                        fig = px.bar(
                            top_surnames,
                            x='Surname',
                            y='Count',
                            title="Top 10 Surnames",
                            color='Count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Facility Analysis")
                
                if not df.empty:
                    if 'FACILITY' in df.columns:
                        # Visits by Facility
                        facility_visits = df['FACILITY'].value_counts().reset_index()
                        facility_visits.columns = ['Facility', 'Visits']
                        
                        fig = px.bar(
                            facility_visits,
                            x='Facility',
                            y='Visits',
                            title="Total Visits by Facility",
                            color='Visits',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Medical Aid by Facility
                        if 'MEDICAL AID' in df.columns:
                            st.subheader("Medical Aid Distribution by Facility")
                            
                            medical_aid_facility = df.groupby(['FACILITY', 'MEDICAL AID']).size().reset_index(name='Count')
                            
                            fig = px.sunburst(
                                medical_aid_facility,
                                path=['FACILITY', 'MEDICAL AID'],
                                values='Count',
                                title="Medical Aid Distribution Across Facilities",
                                color_discrete_sequence=px.colors.sequential.Blues_r
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Gender Distribution by Facility
                        if 'SEX' in df.columns:
                            st.subheader("Gender Distribution by Facility")
                            
                            gender_facility = df.groupby(['FACILITY', 'SEX']).size().reset_index(name='Count')
                            
                            fig = px.bar(
                                gender_facility,
                                x='FACILITY',
                                y='Count',
                                color='SEX',
                                barmode='group',
                                title="Gender Distribution by Facility",
                                color_discrete_map={'Female': '#2a5298', 'Male': '#1e3c72'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Branch Analysis
                    if 'BranchID' in df.columns:
                        st.subheader("Branch Analysis")
                        
                        branch_visits = df['BranchID'].value_counts().reset_index()
                        branch_visits.columns = ['Branch', 'Visits']
                        
                        fig = px.pie(
                            branch_visits,
                            values='Visits',
                            names='Branch',
                            title="Visits by Branch",
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Temporal Trends")
                
                if not df.empty and 'VISIT_DATE' in df.columns and pd.api.types.is_datetime64_any_dtype(df['VISIT_DATE']):
                    # Daily visits
                    daily_visits = df.resample('D', on='VISIT_DATE').size().reset_index(name='Count')
                    
                    fig = px.line(
                        daily_visits,
                        x='VISIT_DATE',
                        y='Count',
                        title="Daily Visits Over Time",
                        color_discrete_sequence=['#1e3c72']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Weekly visits
                    weekly_visits = df.resample('W', on='VISIT_DATE').size().reset_index(name='Count')
                    
                    fig = px.line(
                        weekly_visits,
                        x='VISIT_DATE',
                        y='Count',
                        title="Weekly Visits Over Time",
                        color_discrete_sequence=['#1e3c72']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly visits
                    monthly_visits = df.resample('M', on='VISIT_DATE').size().reset_index(name='Count')
                    
                    fig = px.line(
                        monthly_visits,
                        x='VISIT_DATE',
                        y='Count',
                        title="Monthly Visits Over Time",
                        color_discrete_sequence=['#1e3c72']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Hourly visits (if time component exists)
                    try:
                        df['VISIT_HOUR'] = df['VISIT_DATE'].dt.hour
                        hourly_visits = df['VISIT_HOUR'].value_counts().sort_index().reset_index()
                        hourly_visits.columns = ['Hour', 'Count']
                        
                        fig = px.bar(
                            hourly_visits,
                            x='Hour',
                            y='Count',
                            title="Visits by Hour of Day",
                            color_discrete_sequence=['#1e3c72']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                    
                    # Day of week visits
                    df['DAY_OF_WEEK'] = df['VISIT_DATE'].dt.day_name()
                    dow_visits = df['DAY_OF_WEEK'].value_counts().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ]).reset_index()
                    dow_visits.columns = ['Day of Week', 'Count']
                    
                    fig = px.bar(
                        dow_visits,
                        x='Day of Week',
                        y='Count',
                        title="Visits by Day of Week",
                        color_discrete_sequence=['#1e3c72']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.subheader("Detailed Analysis")
                
                if not df.empty:
                    # Correlation heatmap (for numerical data)
                    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numerical_cols) > 1:
                        st.subheader("Correlation Heatmap")
                        
                        corr_matrix = df[numerical_cols].corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Correlation Between Numerical Variables",
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Medical Aid vs Gender
                    if 'MEDICAL AID' in df.columns and 'SEX' in df.columns:
                        st.subheader("Medical Aid by Gender")
                        
                        medical_gender = df.groupby(['MEDICAL AID', 'SEX']).size().reset_index(name='Count')
                        
                        fig = px.bar(
                            medical_gender,
                            x='MEDICAL AID',
                            y='Count',
                            color='SEX',
                            barmode='group',
                            title="Medical Aid Distribution by Gender",
                            color_discrete_map={'Female': '#2a5298', 'Male': '#1e3c72'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Age Distribution by Medical Aid
                    if 'AGE' in df.columns and 'MEDICAL AID' in df.columns:
                        st.subheader("Age Distribution by Medical Aid")
                        
                        fig = px.box(
                            df,
                            x='MEDICAL AID',
                            y='AGE',
                            color='MEDICAL AID',
                            title="Age Distribution by Medical Aid",
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Patient Visit Frequency
                    if 'PATIENT ID' in df.columns:
                        st.subheader("Patient Visit Frequency")
                        
                        patient_visits = df['PATIENT ID'].value_counts().reset_index()
                        patient_visits.columns = ['Patient ID', 'Visit Count']
                        
                        fig = px.histogram(
                            patient_visits,
                            x='Visit Count',
                            nbins=20,
                            title="Distribution of Visit Counts per Patient",
                            color_discrete_sequence=['#1e3c72']
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Data View Page
        elif st.session_state.current_page == "Data View":
            st.title("📋 Data View")
            
            # Search functionality
            with st.expander("🔍 Search and Filter", expanded=True):
                search_col1, search_col2 = st.columns([3, 1])
                
                with search_col1:
                    search_term = st.text_input("Search across all columns")
                
                with search_col2:
                    rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100])
                
                if search_term:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                    df_display = df[mask]
                else:
                    df_display = df.copy()
            
            # Sort functionality
            with st.expander("🔢 Sort Data", expanded=False):
                sort_col1, sort_col2 = st.columns(2)
                
                with sort_col1:
                    sort_column = st.selectbox("Sort by", df_display.columns)
                
                with sort_col2:
                    sort_direction = st.radio("Direction", ["Ascending", "Descending"])
                
                if sort_column:
                    df_display = df_display.sort_values(
                        by=sort_column,
                        ascending=(sort_direction == "Ascending")
                    )
            
            # Display data with pagination
            st.subheader("Patient Data")
            
            total_pages = (len(df_display) // rows_per_page) + 1
            page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page_number - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            
            st.dataframe(df_display.iloc[start_idx:end_idx], height=600)
            
            # Export functionality
            st.markdown("---")
            st.subheader("📤 Export Data")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                export_format = st.selectbox("Export format", ["CSV", "Excel"])
            
            with export_col2:
                export_filename = st.text_input("Filename", "patient_data_export")
            
            with export_col3:
                st.markdown("")
                st.markdown("")
                if st.button("Export Data"):
                    if export_format == "CSV":
                        csv = df_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{export_filename}.csv",
                            mime="text/csv"
                        )
                    else:
                        excel = df_display.to_excel(index=False, engine='openpyxl')
                        st.download_button(
                            label="Download Excel",
                            data=excel,
                            file_name=f"{export_filename}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        # About Page
        elif st.session_state.current_page == "About":
            st.title("ℹ️ About Corporate 24 Healthcare Dashboard")
            
            st.markdown("""
                ### Dashboard Overview
                This interactive dashboard provides comprehensive analytics and visualization capabilities 
                for Corporate 24 Healthcare patient data.
                
                ### Features
                - **Data Visualization**: Interactive charts and graphs to analyze patient demographics, visit patterns, and facility utilization.
                - **Data Exploration**: Filter, search, and sort through patient records with ease.
                - **Export Capabilities**: Download filtered data in CSV or Excel format.
                
                ### Data Columns
                The dashboard is designed to work with datasets containing the following columns:
                - ENCOUNTER ID
                - PATIENT ID
                - FIRST NAME
                - SURNAME
                - SEX
                - TELEPHONE
                - DATE OF BIRTH
                - MEDICAL AID
                - VISIT_DATE
                - OBSERVED
                - FACILITY
                - BranchID
                
                ### Instructions
                1. Upload your dataset using the sidebar.
                2. Navigate between visualization and data view tabs.
                3. Use filters to focus on specific subsets of data.
                4. Export data or visualizations as needed.
                
                ### Contact
                For support or feature requests, please contact the Corporate 24 IT department.
            """)
    
    else:
        st.title("🏥 Corporate 24 Healthcare Dashboard")
        st.markdown("""
            Welcome to the Corporate 24 Healthcare Analytics Dashboard. 
            
            Please upload a dataset or select from available datasets in the sidebar to begin.
            
            ### Getting Started
            1. **Upload Data**: Use the file uploader in the sidebar to upload your patient data (CSV or Excel format).
            2. **Select Data**: Choose from previously uploaded datasets in the dropdown.
            3. **Explore**: Once data is loaded, navigate between visualization and data view tabs.
            
            ### Sample Data Format
            The dashboard expects data with the following columns:
            - ENCOUNTER ID
            - PATIENT ID
            - FIRST NAME
            - SURNAME
            - SEX
            - TELEPHONE
            - DATE OF BIRTH
            - MEDICAL AID
            - VISIT_DATE
            - OBSERVED
            - FACILITY
            - BranchID
        """)

if __name__ == "__main__":
    main()