# VIVA Q&A — Customer Feedback Analysis Agent
## Agentic AI & Automation — B.Tech Mini Project

---

### Q1. What is Agentic AI?
**A:** Agentic AI refers to AI systems that can autonomously perceive an environment, make decisions, call tools or external services, and complete multi-step tasks without constant human intervention. Unlike traditional ML models that map input → output, an AI agent can plan, reason, and adapt across multiple steps. Our system is agentic because it observes the input, decides which tools to call, sequences them logically, aggregates results, and generates outputs — all autonomously.

---

### Q2. Why is this system called an "Agent" and not just a pipeline?
**A:** A traditional pipeline is fixed: Step A → Step B → Step C. Our system qualifies as an agent because:
1. It **observes** the input (data structure, available columns)
2. It **decides** which tools are relevant (e.g., trend analysis only if date exists, rating analysis only if rating exists)
3. It **orchestrates** multiple specialized tools
4. It **synthesizes** results across tools into insights and recommendations
5. It maintains an **activity log** showing what it did and why
6. It can handle **different types of input** (single text or dataset)

---

### Q3. What is the difference between this system and simple sentiment analysis?
**A:** Simple sentiment analysis classifies text as positive/negative. Our agent goes far beyond:

| Capability | Simple Sentiment | Our Agent |
|------------|-----------------|-----------|
| Sentiment | ✅ | ✅ |
| Emotion Detection | ❌ | ✅ |
| Topic Classification | ❌ | ✅ |
| Issue Detection | ❌ | ✅ |
| Recurring Issue Clustering | ❌ | ✅ |
| Priority Scoring | ❌ | ✅ |
| Business Insights | ❌ | ✅ |
| Recommendations | ❌ | ✅ |
| PDF Report Generation | ❌ | ✅ |

---

### Q4. How does the agent orchestrate its tools?
**A:** The `CustomerFeedbackAgent` class in `agents/customer_feedback_agent.py` orchestrates tools sequentially. It:
1. Accepts input (dataframe + column names)
2. Calls each tool in order, passing the dataframe and collecting results
3. Each tool adds new columns to the dataframe (e.g., sentiment, emotion, topic)
4. Later tools can use results from earlier tools (e.g., priority engine uses sentiment + emotion + issue)
5. Logs each step's action, duration, and status to an activity log

---

### Q5. How are recurring issues detected?
**A:** Using a two-stage approach:
1. **TF-IDF Vectorization**: Converts all feedback texts into numerical vectors capturing important terms
2. **DBSCAN Clustering**: Groups semantically similar vectors using density-based clustering (no need to specify number of clusters)
3. **Labeling**: Each cluster is labeled by finding the most frequent keyword-category match
4. **Filtering**: Only clusters with ≥ 3 occurrences are considered "recurring"
5. **Fallback**: If sklearn is unavailable, keyword-based grouping is used

---

### Q6. How is priority calculated? What factors are used?
**A:** The Priority Engine uses a multi-factor weighted scoring system:

| Factor | Points |
|--------|--------|
| Negative sentiment | +3 |
| Angry/Frustrated emotion | +3 to +4 |
| Complaint detected | +2 |
| High-impact issue type | +2 |
| Urgency keywords (urgent, ASAP, etc.) | +1 per keyword |
| Business impact signals (refund, cancel, etc.) | +1 per signal |
| Very low rating (≤2) | +2 |

**Classification**:
- Score ≥ 7 → HIGH
- Score 4-6 → MEDIUM
- Score < 4 → LOW

Priority reasons are also stored and displayed to the user for full transparency.

---

### Q7. How are recommendations generated?
**A:** The recommendation engine uses a two-step process:
1. **Issue Mapping**: Each detected issue type (e.g., "Delivery Delay") maps to a predefined, domain-expert-curated recommendation template
2. **Evidence Population**: The template is populated with actual data (occurrences, percentage, sentiment) so the recommendation is evidence-specific, not generic
3. **Prioritization**: Recommendations are sorted by issue priority
4. **KPI and Department**: Each recommendation includes a measurable KPI target and responsible department

---

### Q8. How does the system handle unseen/new types of feedback?
**A:** The system is designed to be robust to new input:
- Topic detection falls back to "Other" for unrecognized topics
- Issue detection falls back to "General Feedback" for unmatched patterns
- Sentiment analysis works on any English text via VADER's lexical approach
- Recurring issue clustering is unsupervised — it discovers patterns without predefined labels
- The priority engine operates on any combination of sentiment, emotion, and keyword signals

---

### Q9. What happens if the date or rating column is missing?
**A:** The system gracefully handles missing optional columns:
- **Date missing**: Trend analysis is skipped; a message is shown: "Trend analysis unavailable because no date field was provided."
- **Rating missing**: Rating analysis section is hidden
- The agent actively checks column availability before calling relevant tools, demonstrating intelligent decision-making

---

### Q10. What NLP models and techniques are used?
**A:**
- **Sentiment**: VADER (Valence Aware Dictionary and sEntiment Reasoner) — a lexicon + rule-based model tuned for social media text; TextBlob as backup
- **Emotion**: Rule-based keyword matching with intensity modifiers and negation handling
- **Topic Detection**: Keyword matching with multi-word phrase scoring
- **Recurring Issues**: TF-IDF (Term Frequency-Inverse Document Frequency) for vectorization + DBSCAN for clustering
- **Issue Detection**: Regex pattern matching with multiple patterns per issue type

---

### Q11. Why were these technologies selected?
**A:**
- **Streamlit**: Rapid Python dashboard, no frontend knowledge needed, perfect for demos
- **VADER**: Designed for social media/review text, no training required, open-source
- **sklearn DBSCAN**: Best for clustering unknown number of groups, handles noise well
- **ReportLab**: Professional PDF generation with full layout control
- **Plotly**: Interactive charts that work natively in Streamlit
- **No paid API**: All tools work locally, making the project reproducible and accessible

---

### Q12. What are the limitations of this system?
**A:**
1. **Language**: Currently optimized for English text
2. **Sarcasm**: VADER and keyword-based approaches may misclassify sarcastic text
3. **Context**: Individual sentences analyzed in isolation — no cross-sentence context
4. **Domain adaptability**: Keyword lists may need tuning for highly specialized domains (e.g., medical, legal)
5. **Scale**: For very large datasets (>100k records), vectorization may be slow
6. **LLM**: Unlike GPT-based systems, insight quality is limited by rule-based logic

---

### Q13. How can this be deployed in a real company?
**A:**
1. **Cloud Deployment**: Deploy Streamlit app on AWS/GCP/Azure or Streamlit Cloud
2. **API Integration**: Convert agent to a REST API (FastAPI) for integration with CRM systems
3. **Database**: Connect to a PostgreSQL/BigQuery database instead of file uploads
4. **Automation**: Schedule daily runs using Apache Airflow or Cloud Scheduler
5. **LLM Enhancement**: Integrate GPT-4/Gemini for better insight generation with API fallback
6. **Authentication**: Add OAuth for multi-user access control

---

### Q14. How is data quality ensured?
**A:** The Data Cleaning Tool performs:
1. Null removal — drops rows where feedback is missing
2. Deduplication — removes exact duplicate feedback texts
3. Text normalization — strips extra whitespace, special characters
4. Short text filtering — removes feedback < 3 words (noise)
5. Reports a data quality summary visible to the user

---

### Q15. What makes the AI Insights different from hard-coded statements?
**A:** Every insight is computed from actual data:
- Negative percentage → computed from `df["sentiment"].value_counts()`
- Top topic → computed from `df["topic"].value_counts().index[0]`
- Recurring issue label → from clustering output with actual count
- Trend insight → from comparing first vs second half of time-series data
- Values change every time the dataset changes — zero hard-coding

---

### Q16. What is the agentic loop in your system?
**A:** The agentic loop is:
```
Observe → Plan → Act → Observe → Plan → Act → ...
```
1. Observe: Read input data, inspect structure
2. Plan: Determine which tools apply (date? rating?)
3. Act: Call Data Cleaner → read output
4. Observe: Check cleaned data size
5. Plan: Proceed to Sentiment (always applies)
6. Act: Call Sentiment Tool → collect results
7. ... repeat for all tools
8. Synthesize: Combine all results into insights and report

---

### Q17. How does the system handle different file formats?
**A:** The Upload Dataset page accepts CSV and Excel files. Pandas handles both:
- `.csv` → `pd.read_csv()`
- `.xlsx`, `.xls` → `pd.read_excel()`

Column detection is automatic with manual override available.

---

### Q18. What is the purpose of the Activity Log page?
**A:** The Activity Log shows the full transparent agent workflow:
- Every tool called with timestamp and duration
- What was found at each step
- Success/error status for each step
- This is critical for demonstrating "agentic" behavior — the system explains its own actions

---

### Q19. How does the recommendation engine ensure recommendations are actionable?
**A:** Each recommendation includes:
1. **Issue**: The specific problem being addressed
2. **Evidence**: Number of occurrences and percentage of total feedback
3. **Action**: Concrete business action to take
4. **Details**: Step-by-step implementation guidance
5. **KPI Target**: A measurable success metric (e.g., "Reduce complaints by 30% within 60 days")
6. **Department**: The team responsible for implementation

---

### Q20. How does your project demonstrate the concept taught in Agentic AI & Automation?
**A:** The project demonstrates:
1. **Agent Architecture**: Orchestrator + specialized tools pattern
2. **Autonomous Decision-Making**: Tool selection based on available data
3. **Tool Use**: 11 distinct tools called and coordinated
4. **Observe-Plan-Act cycle**: Visible in the activity log
5. **Human-AI Collaboration**: User provides data, agent provides complete analysis
6. **Transparency**: Full activity log, priority reasoning, evidence-based outputs
7. **Automation**: Entire analysis pipeline runs without human intervention after data upload

---

*Document prepared for B.Tech Viva — Agentic AI & Automation (Flexi Credit Course)*
