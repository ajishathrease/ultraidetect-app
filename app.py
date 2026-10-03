import os
import sqlite3
import hashlib
from datetime import datetime
import io

import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Imports for Grad-CAM (XAI)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set Page Config
st.set_page_config(page_title="UltrAIDetect - AI Breast Ultrasound Diagnostic Platform", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for UI styling & Centered Login Box
st.markdown("""
    <style>
        .auth-container {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 30px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            margin-top: 20px;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
DB_FILE = "patient_history.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            tissue_composition TEXT,
            lesion_shape TEXT,
            lesion_margin TEXT,
            echogenicity TEXT,
            posterior_features TEXT,
            prediction TEXT,
            normal_prob REAL,
            benign_prob REAL,
            malignant_prob REAL,
            clinician_username TEXT,
            timestamp DATETIME
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            full_name TEXT
        )
    ''')
    c.execute("SELECT count(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                  ("pathologist1", hash_password("medical2026"), "Dr. Primary Pathologist"))
    conn.commit()
    conn.close()

init_db()

def register_user(username, password, full_name):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                  (username, hash_password(password), full_name))
        conn.commit()
        conn.close()
        return True, "Account registered successfully! You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."

def verify_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash, full_name FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        return True, row[1]
    return False, None

def save_record(patient_id, name, age, tissue, shape, margin, echo, post, pred, p_norm, p_ben, p_mal, clinician):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO patient_records 
        (patient_id, patient_name, age, tissue_composition, lesion_shape, lesion_margin, echogenicity, posterior_features, prediction, normal_prob, benign_prob, malignant_prob, clinician_username, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, name, age, tissue, shape, margin, echo, post, pred, p_norm, p_ben, p_mal, clinician, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_history(username_filter=None):
    conn = sqlite3.connect(DB_FILE)
    if username_filter:
        query = "SELECT * FROM patient_records WHERE clinician_username = ? ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn, params=(username_filter,))
    else:
        query = "SELECT * FROM patient_records ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ==========================================
# 2. MODEL ARCHITECTURE & LOADERS
# ==========================================
class MultimodalBreastCancerNet(nn.Module):
    def __init__(self, num_clinical_features):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.tabular_backbone = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, image, tabular_data):
        img_feats = torch.flatten(self.image_backbone(image), 1)
        tab_feats = self.tabular_backbone(tabular_data)
        return self.classifier(torch.cat((img_feats, tab_feats), dim=1))

class ImageOnlyModelWrapper(nn.Module):
    def __init__(self, full_model, clinical_tensor):
        super().__init__()
        self.full_model = full_model
        self.clinical_tensor = clinical_tensor

    def forward(self, x):
        return self.full_model(x, self.clinical_tensor)

@st.cache_resource
def load_model_and_features():
    with open("feature_columns.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines() if line.strip()]
    model = MultimodalBreastCancerNet(num_clinical_features=len(feature_cols))
    model.load_state_dict(torch.load("breast_cancer_multimodal_3class.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, feature_cols

model, feature_cols = load_model_and_features()

# ==========================================
# 3. PDF REPORT GENERATOR
# ==========================================
def generate_pdf_report(patient_id, name, age, pred_class, probs, orig_img, cam_img, shap_fig, clinician):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A365D'), spaceAfter=12)
    story.append(Paragraph("<b>CLINICAL MULTIMODAL DIAGNOSTIC REPORT</b>", title_style))
    story.append(Spacer(1, 10))

    data_summary = [
        [Paragraph(f"<b>Patient ID:</b> {patient_id}"), Paragraph(f"<b>Patient Name:</b> {name}")],
        [Paragraph(f"<b>Age:</b> {age}"), Paragraph(f"<b>Attending Clinician:</b> {clinician}")],
        [Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}"), Paragraph(f"<b>Assessment:</b> <font color='red'><b>{pred_class}</b></font>")]
    ]
    t_summary = Table(data_summary, colWidths=[260, 260])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Model Confidence Scores</b>", styles['Heading2']))
    prob_data = [
        ["Class Normal", "Class Benign", "Class Malignant"],
        [f"{probs[0]*100:.2f}%", f"{probs[1]*100:.2f}%", f"{probs[2]*100:.2f}%"]
    ]
    t_prob = Table(prob_data, colWidths=[170, 170, 170])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_prob)
    story.append(Spacer(1, 15))

    # Convert Images
    orig_buf = io.BytesIO()
    cam_buf = io.BytesIO()
    shap_buf = io.BytesIO()
    
    orig_img.save(orig_buf, format='PNG')
    Image.fromarray(cam_img).save(cam_buf, format='PNG')
    shap_fig.savefig(shap_buf, format='PNG', bbox_inches='tight')
    
    orig_buf.seek(0)
    cam_buf.seek(0)
    shap_buf.seek(0)

    story.append(Paragraph("<b>Visual Explainability (Grad-CAM Target Focus)</b>", styles['Heading2']))
    img_table_data = [
        [RLImage(orig_buf, width=220, height=180), RLImage(cam_buf, width=220, height=180)],
        [Paragraph("Original Scan", styles['Normal']), Paragraph(f"Grad-CAM Heatmap ({pred_class})", styles['Normal'])]
    ]
    t_imgs = Table(img_table_data, colWidths=[260, 260])
    t_imgs.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(t_imgs)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Clinical Tabular Attribution Analysis</b>", styles['Heading2']))
    story.append(RLImage(shap_buf, width=480, height=180))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ==========================================
# 4. SESSION STATE INITIALIZATION
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'full_name' not in st.session_state:
    st.session_state['full_name'] = ""
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"

# ==========================================
# 5. AUTHENTICATION SCREEN
# ==========================================
if not st.session_state['logged_in']:
    st.markdown("<br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 1.8, 1])
    
    with center_col:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>🔬 Ultr<span style='color: #3182ce;'>AID</span>etect - AI Breast Ultrasound Diagnostic Platform</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b;'>Multimodal Diagnostic Workbench</p>", unsafe_allow_html=True)
        
        auth_tab1, auth_tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with auth_tab1:
            login_user = st.text_input("Username", key="login_user")
            login_pwd = st.text_input("Password", type="password", key="login_pwd")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Log In", use_container_width=True):
                if login_user and login_pwd:
                    is_valid, full_name = verify_user(login_user, login_pwd)
                    if is_valid:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = login_user
                        st.session_state['full_name'] = full_name if full_name else login_user
                        st.session_state['current_page'] = "Home"
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please fill in both fields.")

        with auth_tab2:
            reg_fullname = st.text_input("Full Name", placeholder="e.g., Dr. Jane Smith", key="reg_name")
            reg_user = st.text_input("Choose Username", key="reg_user")
            reg_pwd = st.text_input("Choose Password", type="password", key="reg_pwd")
            reg_pwd_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd_conf")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register Account", use_container_width=True):
                if not reg_fullname or not reg_user or not reg_pwd:
                    st.warning("Please fill out all mandatory fields.")
                elif reg_pwd != reg_pwd_confirm:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(reg_user, reg_pwd, reg_fullname)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Navigation Bar
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([2, 1, 1, 1, 1])
    with nav_col1:
        st.markdown(f"#### 🩺 **{st.session_state['full_name']}**")
    with nav_col2:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state['current_page'] = "Home"
            st.rerun()
    with nav_col3:
        if st.button("🔬 New Analysis", use_container_width=True):
            st.session_state['current_page'] = "New Diagnostic Analysis"
            st.rerun()
    with nav_col4:
        if st.button("📋 Patient History", use_container_width=True):
            st.session_state['current_page'] = "Patient History Log"
            st.rerun()
    with nav_col5:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.session_state['full_name'] = ""
            st.rerun()

    st.markdown("---")

    # ==========================================
    # PAGE 1: HOME LANDING PAGE WITH TUTORIAL
    # ==========================================
    if st.session_state['current_page'] == "Home":
        st.markdown("# 🏫 Ultr<span style='color: #3182ce;'>AID</span>etect - AI Breast Ultrasound Diagnostic Platform", unsafe_allow_html=True)
        st.markdown("Automated Multimodal Breast Cancer Assessment & Multi-Class Explainable AI (XAI) Workbench.")
        
        col_banner1, col_banner2 = st.columns([2, 1])
        with col_banner1:
            st.info("""
            ### Welcome to the Multimodal Diagnostic Workstation
            This portal provides **interactive dual-modality explainability**:
            * **Grad-CAM Visual Heatmaps**: Toggle and inspect target regions for **Normal**, **Benign**, and **Malignant** classes.
            * **Tabular Feature Attribution**: Evaluate precise positive and negative contributions of BI-RADS clinical attributes.
            """)
            if st.button("🚀 Start Analysis", use_container_width=True):
                st.session_state['current_page'] = "New Diagnostic Analysis"
                st.rerun()

        with col_banner2:
            st.metric("System Status", "Online & Ready")
            st.metric("Backbone Model", "ResNet-50 + Tabular MLP")

        st.markdown("---")
        st.subheader("📖 Application Usage Tutorial")
        
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.markdown("""
            #### Step 1: Input Metadata
            * Navigate to the **New Analysis** tab.
            * Enter the patient's ID, Name, and Age.
            * Select all appropriate BI-RADS clinical attributes (Tissue Composition, Lesion Shape, Margin, Echogenicity, Posterior Features).
            """)
        with t_col2:
            st.markdown("""
            #### Step 2: Upload Scan
            * Upload a high-resolution ultrasound image file (`.png`, `.jpg`, `.jpeg`).
            * Review the visual scan preview in the right column.
            * Click **Process Scan & Run Assessment**.
            """)
        with t_col3:
            st.markdown("""
            #### Step 3: Evaluate & Export
            * Review class probabilities (**Normal**, **Benign**, **Malignant**).
            * Inspect the heatmaps generated by **Grad-CAM Visual XAI** across classes.
            * Analyze **Tabular Attribution** scores and download the PDF diagnostic report.
            """)

    # ==========================================
    # PAGE 2: NEW DIAGNOSTIC ANALYSIS (WITH DUAL XAI)
    # ==========================================
    elif st.session_state['current_page'] == "New Diagnostic Analysis":
        st.title("🔬 Multimodal Diagnostic Workstation")
        st.markdown("Integrated Assessment with Interactive Grad-CAM & Tabular Attribution Explainability")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("1. Patient Demographic & Metadata")
            p_id = st.text_input("Patient ID", value="", placeholder="Enter Patient ID")
            p_name = st.text_input("Patient Name", value="", placeholder="Enter Patient Name")
            age = st.slider("Patient Age", 18, 90, 45)
            
            tissue_comp = st.selectbox("Tissue Composition", ["Select", "homogeneous: fat", "homogeneous: fibroglandular", "heterogeneous: predominant fat", "heterogeneous: predominant fibroglandular", "None/Normal"])
            shape = st.selectbox("Lesion Shape", ["Select", "oval", "round", "irregular", "None/Normal"])
            margin = st.selectbox("Lesion Margin", ["Select", "circumscribed", "not circumscribed", "None/Normal"])
            echogenicity = st.selectbox("Echogenicity", ["Select", "anechoic", "hyperechoic", "complex cystic and solid", "hypoechoic", "isoechoic", "heterogeneous", "None/Normal"])
            posterior = st.selectbox("Posterior Features", ["Select", "no posterior features", "shadowing", "enhancement", "combined", "None/Normal"])

            st.subheader("2. Ultrasound Image Upload")
            uploaded_file = st.file_uploader("Upload Ultrasound Scan", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            with col_right:
                st.subheader("Scan Preview")
                st.image(image, use_container_width=True)

            if st.button("🚀 Process Scan & Run Assessment", use_container_width=True):
                if not p_id.strip() or not p_name.strip():
                    st.error("⚠️ Please fill in both Patient ID and Patient Name.")
                elif "Select" in [tissue_comp, shape, margin, echogenicity, posterior]:
                    st.error("⚠️ Please select valid options for all clinical dropdown fields.")
                else:
                    # Input Preprocessing
                    img_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img_tensor = img_transform(image).unsqueeze(0)

                    norm_age = (age - 18) / (90 - 18)
                    user_inputs = {
                        'norm_age': norm_age,
                        f'Tissue_composition_{tissue_comp}': 1,
                        f'Shape_{shape}': 1,
                        f'Margin_{margin}': 1,
                        f'Echogenicity_{echogenicity}': 1,
                        f'Posterior_features_{posterior}': 1
                    }

                    clinical_vec = np.zeros(len(feature_cols), dtype=np.float32)
                    for idx, col_name in enumerate(feature_cols):
                        if col_name in user_inputs:
                            clinical_vec[idx] = user_inputs[col_name]
                    clinical_tensor = torch.tensor(clinical_vec).unsqueeze(0)

                    # Model Forward Pass
                    with torch.no_grad():
                        outputs = model(img_tensor, clinical_tensor)
                        probs = torch.softmax(outputs, dim=1).numpy()[0]

                    class_names = ["Normal", "Benign", "Malignant"]
                    pred_idx = int(np.argmax(probs))
                    pred_name = class_names[pred_idx]

                    # Save SQLite Record
                    save_record(p_id, p_name, age, tissue_comp, shape, margin, echogenicity, posterior, pred_name, float(probs[0]), float(probs[1]), float(probs[2]), st.session_state['username'])
                    st.success("✅ Diagnostic results logged to SQLite database!")

                    # Results Presentation
                    st.markdown("---")
                    st.header("Assessment Probabilities")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Normal", f"{probs[0]*100:.2f}%")
                    c2.metric("Benign", f"{probs[1]*100:.2f}%")
                    c3.metric("Malignant", f"{probs[2]*100:.2f}%")

                    st.markdown("---")
                    st.header("💡 Multimodal Explainable AI (XAI) Suite")

                    # ==========================================
                    # FEATURE 1: INTERACTIVE CLASS GRAD-CAM
                    # ==========================================
                    st.subheader("1. Interactive Grad-CAM Heatmap Focus")
                    st.write("Select a diagnostic class to view what visual regions influenced that specific prediction:")

                    target_layers = [model.image_backbone[7][-1].conv3]
                    wrapped_model = ImageOnlyModelWrapper(model, clinical_tensor)
                    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
                    rgb_img = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0

                    cam_tabs = st.tabs(["Class: Normal", "Class: Benign", "Class: Malignant"])
                    cam_visualizations = {}

                    for idx, class_label in enumerate(class_names):
                        grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(idx)])[0]
                        viz = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                        cam_visualizations[class_label] = viz
                        
                        with cam_tabs[idx]:
                            v_col1, v_col2 = st.columns(2)
                            with v_col1:
                                st.image(image.resize((224, 224)), caption="Original Ultrasound Scan", use_container_width=True)
                            with v_col2:
                                st.image(viz, caption=f"Grad-CAM Attention ({class_label} Focus)", use_container_width=True)

                    # ==========================================
                    # FEATURE 2: FAST RELIABLE TABULAR ATTRIBUTION
                    # ==========================================
                    st.markdown("---")
                    st.subheader("2. Tabular Feature Attribution (SHAP Analysis)")
                    st.write(f"Contribution of individual clinical attributes toward top prediction (**{pred_name}**):")

                    # Gradient/Weight Attribution Computation for Tabular Input
                    clinical_input_grad = clinical_tensor.clone().detach().requires_grad_(True)
                    logits = model(img_tensor, clinical_input_grad)
                    target_score = logits[0, pred_idx]
                    target_score.backward()
                    
                    grads = clinical_input_grad.grad.detach().numpy()[0]
                    attr_values = grads * clinical_vec  # Feature x Gradient sensitivity

                    # Build Attribution DataFrame
                    df_attr = pd.DataFrame({
                        'Feature': feature_cols,
                        'Attribution': attr_values,
                        'Value': clinical_vec
                    })

                    # Filter non-zero active features for clear visual plotting
                    active_df = df_attr[df_attr['Value'] > 0].sort_values(by='Attribution', key=abs, ascending=True)
                    if active_df.empty:
                        active_df = df_attr.sort_values(by='Attribution', key=abs, ascending=True).tail(5)

                    # Plotting Attribution
                    fig_shap, ax = plt.subplots(figsize=(8, 3.5))
                    colors_list = ['#e53e3e' if x >= 0 else '#3182ce' for x in active_df['Attribution']]
                    ax.barh(active_df['Feature'], active_df['Attribution'], color=colors_list)
                    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                    ax.set_title(f"Clinical Feature Impact on '{pred_name}' Class Prediction", fontsize=10)
                    ax.set_xlabel("Attribution Value (Positive = Increases Confidence, Negative = Decreases)", fontsize=9)
                    plt.tight_layout()

                    st.pyplot(fig_shap)

                    # Generate PDF Report
                    pdf_bytes = generate_pdf_report(
                        p_id, p_name, age, pred_name, probs, 
                        image.resize((224, 224)), 
                        cam_visualizations[pred_name], 
                        fig_shap, 
                        st.session_state['full_name']
                    )
                    
                    st.markdown("---")
                    st.download_button(
                        label="📄 Download Official PDF Clinical Report",
                        data=pdf_bytes,
                        file_name=f"Report_{p_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )

    # ==========================================
    # PAGE 3: SEPARATED PATIENT HISTORY LOG
    # ==========================================
    elif st.session_state['current_page'] == "Patient History Log":
        st.title("📋 Patient Record History")
        st.markdown("Database view of multimodal diagnostic assessments.")
        
        hist_tab1, hist_tab2 = st.tabs(["👤 My Handled Cases", "🌐 All Patient Records"])
        
        with hist_tab1:
            st.subheader(f"Cases Logged by {st.session_state['full_name']}")
            df_personal = fetch_history(username_filter=st.session_state['username'])
            if not df_personal.empty:
                st.dataframe(df_personal, use_container_width=True)
                csv_personal = df_personal.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export My History to CSV", data=csv_personal, file_name=f"my_patient_history_{st.session_state['username']}.csv", mime="text/csv")
            else:
                st.info("You have not processed any patient records yet.")

        with hist_tab2:
            st.subheader("System-Wide Patient Database")
            df_total = fetch_history(username_filter=None)
            if not df_total.empty:
                st.dataframe(df_total, use_container_width=True)
                csv_total = df_total.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export All History to CSV", data=csv_total, file_name="total_patient_history_export.csv", mime="text/csv")
            else:
                st.info("No patient records found in the overall database.")