"""
Interactive Demo App for Editorial Summarization System
Streamlit web interface - OFFLINE MODE (uses local models only)

Author: ZabeeAhmed
Date: 2025-11-19 12:50:15 UTC
"""

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import os

# Disable HuggingFace Hub access (work offline)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Page config
st.set_page_config(
    page_title="Editorial Summarization System",
    page_icon="📰",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_models():
    """Load fine-tuned model only (offline mode)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Only load fine-tuned model (already local)
        model_path = '../models/finetuned_editorial_t5'
        
        if not Path(model_path).exists():
            st.error(f"❌ Model not found at: {model_path}")
            st.info("Please ensure the fine-tuned model is in the correct location.")
            return None
        
        tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_summary(model, tokenizer, text, device, max_length=150):
    """Generate summary"""
    input_text = f"summarize: {text}"
    
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load model
with st.spinner("Loading fine-tuned model..."):
    models = load_models()

if models is None:
    st.stop()

# Header
st.title("📰 Editorial Summarization System")
st.markdown("### AI-Powered Summarization for Indian Newspaper Editorials")
st.markdown("**Author:** ZabeeAhmed | **Model:** Fine-tuned T5-small")

# Sidebar - Project Info
with st.sidebar:
    st.header("📊 Model Performance")
    
    st.metric("ROUGE-1 Score", "0.5713", "+156%")
    st.metric("ROUGE-2 Score", "0.5094", "+273%")
    st.metric("ROUGE-L Score", "0.5407", "+187%")
    
    st.markdown("---")
    
    st.header("🎯 Project Info")
    st.markdown("""
    **Dataset:** 87 Indian newspaper editorials
    
    **Method:** Domain adaptation via fine-tuning
    
    **Base Model:** T5-small
    
    **Training:** 5 epochs, batch size 4
    
    **Validation Loss:** 0.8122
    
    **Improvement:** 273% on ROUGE-2
    """)
    
    st.markdown("---")
    
    st.header("🔗 Links")
    st.markdown("📁 [GitHub Repository](https://github.com/ZabeeAhamed)")
    st.markdown("💼 [LinkedIn](https://linkedin.com/in/zabeeahmed)")

# Main content
tab1, tab2, tab3 = st.tabs(["🚀 Generate Summary", "📝 Examples", "ℹ️ About"])

with tab1:
    st.markdown("### Enter an editorial article to summarize")
    
    # Sample text button
    if st.button("📋 Load Sample Article"):
        sample_text = """With President Droupadi Murmu withholding assent for the Tamil Nadu Admission to Undergraduate Medical Degree Courses Bill 2021, the State faces another setback in its attempt to exempt itself from the NEET-based admission system. Chief Minister M.K. Stalin has convened a meeting to discuss future steps, as this uncertainty impacts students. The Bill was originally passed in 2021 based on Justice A.K. Rajan Committee recommendations, was returned by the Governor and then re-adopted and sent to the President. The delay in assent leaves students in limbo. NEET has been upheld consistently by courts, and the President is under no obligation to approve the Bill. Despite political consensus in Tamil Nadu, legal options are limited, and a quick judicial resolution is unlikely. The editorial concludes that the State should prepare legally and simultaneously support students in NEET preparation."""
        st.session_state['article_input'] = sample_text
    
    # Input text area
    article_input = st.text_area(
        "Paste your article here:",
        height=300,
        value=st.session_state.get('article_input', ''),
        placeholder="Enter a newspaper editorial article (minimum 100 words)...",
        help="Works best with Indian newspaper editorials",
        key='article_text'
    )
    
    # Generate button
    if st.button("🎯 Generate Summary", type="primary"):
        if not article_input or len(article_input.split()) < 50:
            st.error("⚠️ Please enter at least 50 words for meaningful summarization.")
        else:
            with st.spinner("Generating summary..."):
                
                # Display input stats
                word_count = len(article_input.split())
                char_count = len(article_input)
                
                st.markdown(f"**Input:** {word_count} words | {char_count} characters")
                
                st.markdown("---")
                
                # Generate summary
                st.markdown("### 🟢 Generated Summary (Fine-tuned T5)")
                
                try:
                    summary = generate_summary(
                        models['model'],
                        models['tokenizer'],
                        article_input,
                        models['device']
                    )
                    
                    st.success(summary)
                    
                    summary_words = len(summary.split())
                    compression = summary_words / word_count
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Summary Length", f"{summary_words} words")
                    col2.metric("Compression", f"{compression:.1%}")
                    col3.metric("Reduction", f"-{100-compression*100:.0f}%")
                    
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

with tab2:
    st.markdown("### 📝 Sample Summaries")
    
    st.info("💡 **Tip:** Click 'Load Sample Article' in the Generate tab to try a real example!")
    
    # Example articles
    examples = [
        {
            "title": "Tamil Nadu NEET Bill",
            "article": "With President Droupadi Murmu withholding assent for the Tamil Nadu Admission to Undergraduate Medical Degree Courses Bill 2021, the State faces another setback in its attempt to exempt itself from the NEET-based admission system. CM M.K. Stalin has convened a meeting to discuss future steps...",
            "summary": "President Murmu's withholding of assent to Tamil Nadu's anti-NEET Bill prolongs the state's legal struggle. The editorial urges the State to prepare for a legal battle while simultaneously helping students face NEET."
        },
        {
            "title": "India-Sri Lanka Relations",
            "article": "Prime Minister Narendra Modi's visit to Sri Lanka demonstrated the close ties between the two countries. The conferment of Sri Lanka's highest honour and a defense cooperation MoU marked positive strides in bilateral relations...",
            "summary": "PM Modi's Sri Lanka visit strengthened bilateral ties with a defense MoU and reassurances on security. Key discussions on fishermen's conflict and Tamil political settlement were welcomed."
        }
    ]
    
    for idx, example in enumerate(examples, 1):
        with st.expander(f"Example {idx}: {example['title']}"):
            st.markdown("**Article:**")
            st.write(example['article'])
            st.markdown("**Summary:**")
            st.info(example['summary'])

with tab3:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    ## Editorial Summarization System
    
    This project demonstrates **domain adaptation** for automatic text summarization, 
    specifically fine-tuned for Indian newspaper editorials.
    
    ### 🎯 Key Features
    
    - ✅ **Domain-Specific Training:** Fine-tuned on 87 real Indian newspaper editorials
    - ✅ **Massive Improvement:** 156-273% better than baseline T5 model
    - ✅ **Production-Ready:** Complete ML pipeline from OCR to deployment
    - ✅ **Offline Capable:** Works without internet connection
    
    ### 🛠️ Technical Stack
    
    - **Model:** T5-small (60M parameters)
    - **Framework:** PyTorch, Hugging Face Transformers
    - **OCR:** Tesseract for newspaper image processing
    - **Evaluation:** ROUGE metrics
    - **Deployment:** Streamlit web app
    
    ### 📊 Performance Metrics
    
    | Metric | Pretrained | Fine-tuned | Improvement |
    |--------|-----------|------------|-------------|
    | ROUGE-1 | 0.2229 | **0.5713** | **+156%** 📈 |
    | ROUGE-2 | 0.1364 | **0.5094** | **+273%** 🔥 |
    | ROUGE-L | 0.1886 | **0.5407** | **+187%** ⭐ |
    
    ### 🎓 Methodology
    
    1. **Data Collection:** OCR extraction from 196 newspaper images
    2. **Data Processing:** Quality validation and filtering → 87 pairs
    3. **Model Training:** Fine-tuning T5-small for 5 epochs
    4. **Evaluation:** ROUGE score comparison
    5. **Deployment:** Interactive web interface
    
    ### ⚠️ Limitations
    
    - OCR quality affects input (newspaper scans have inherent noise)
    - Optimized for Indian editorial style
    - English-only (no multilingual support yet)
    
    ### 👨‍💻 Author
    
    **ZabeeAhmed**
    - GitHub: [ZabeeAhamed](https://github.com/ZabeeAhamed)
    - Date: November 2025
    
    ### 📚 Technical Details
    
    - **Training Time:** ~25 seconds on CUDA GPU
    - **Model Size:** ~250MB
    - **Inference Speed:** ~0.5s per summary
    - **Max Input:** 512 tokens (~400 words)
    - **Output Range:** 30-150 words
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    Made with ❤️ by ZabeeAhamed | Powered by T5 & Streamlit<br>
    🚀 Fine-tuned Model | 📊 273% ROUGE-2 Improvement
    </div>
    """,
    unsafe_allow_html=True
)