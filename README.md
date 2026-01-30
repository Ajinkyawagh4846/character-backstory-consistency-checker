# Character Backstory Validator 🎯

AI-powered system that verifies if fictional character backstories are logically consistent with events in literary novels. Uses retrieval-augmented generation (RAG), multi-claim reasoning, and causal analysis.

**🏆 KDSH 2026 Hackathon Submission**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Problem

Given:
- A **character backstory** (fictional background story)
- A **novel** (100,000+ words)

Determine:
- ✅ **Consistent**: Backstory logically fits with the character's actions and story events
- ❌ **Contradict**: Backstory conflicts with facts or makes future events implausible

### Example

**Backstory:** "Harry Potter was loved by his aunt and uncle and had a happy childhood."

**Novel:** The Dursleys mistreated Harry, made him sleep in a cupboard...

**Decision:** ❌ **CONTRADICT** - Direct contradiction with established narrative

---

## 🧠 Solution Architecture

### Multi-Stage Reasoning Pipeline
```
Novel (100k+ words)          Backstory
        ↓                         ↓
    Chunk & Index          Extract Claims
        ↓                         ↓
   ChromaDB Vector Store    [Claim 1, Claim 2, ...]
        ↓                         ↓
        └─────→ Semantic Retrieval ←─────┘
                       ↓
              Relevant Passages
                       ↓
           Gemini 2.5 Flash Analysis
                       ↓
        Consistency Check per Claim
                       ↓
              Aggregate Results
                       ↓
         Final: consistent/contradict
```

### Key Innovations

🔍 **Claim Decomposition**: Breaks complex backstories into atomic, verifiable claims
```
"John grew up poor, his father was abusive, he learned to fight"
→ Claim 1: "John grew up poor"
→ Claim 2: "John's father was abusive"  
→ Claim 3: "John learned to fight"
```

🎯 **Causal Reasoning**: Not just contradiction detection—evaluates if past events make future actions plausible

📊 **Confidence Scoring**: Weighted decisions based on evidence strength and claim agreement

🔄 **Multi-Hop Retrieval**: Uses vector similarity to find relevant passages across entire novel

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Text Embedding 004 |
| **Vector Database** | ChromaDB |
| **Language** | Python 3.8+ |
| **Key Libraries** | `google-generativeai`, `chromadb`, `pandas`, `tqdm` |

---

## 📂 Project Structure
```
character-backstory-validator/
├── code/
│   ├── explore_data.py          # Dataset analysis & statistics
│   ├── retriever.py              # Novel search engine (RAG)
│   ├── consistency_checker.py   # LLM reasoning & claim analysis
│   └── main.py                   # Complete pipeline
├── books/
│   ├── The Count of Monte Cristo.txt
│   └── In Search of the Castaways.txt
├── results/
│   ├── dataset_summary.txt      # Data exploration results
│   └── submission.csv            # Final predictions
├── train.csv                     # 80 labeled examples
├── test.csv                      # 60 test cases
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Gemini API key ([Get one free](https://aistudio.google.com/app/apikey))

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/character-backstory-validator.git
cd character-backstory-validator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```

### Usage

**Option 1: Full Pipeline (with validation)**
```bash
python code/main.py
# Answer 'y' to run validation on 5 training samples
# Answer 'y' to process all 60 test cases
```

**Option 2: Data Exploration Only**
```bash
python code/explore_data.py
# Generates results/dataset_summary.txt
```

---

## 📊 Dataset

- **Training Set**: 80 labeled examples (consistent/contradict)
- **Test Set**: 60 unlabeled cases
- **Novels**: 2 classic books
  - *The Count of Monte Cristo* (2.66 MB, 186 chunks)
  - *In Search of the Castaways* (825 KB, 56 chunks)
- **Characters**: 6 unique characters
- **Label Distribution**: 64% consistent, 36% contradict

---

## 🎯 How It Works (Detailed)

### Step 1: Novel Indexing
```python
# retriever.py
1. Load novel text
2. Split into 3000-word overlapping chunks
3. Generate embeddings using Gemini Text Embedding 004
4. Store in ChromaDB vector database
```

### Step 2: Claim Extraction
```python
# consistency_checker.py
1. Send backstory to Gemini 2.5 Flash
2. Extract 5-7 atomic claims
3. Each claim is specific and verifiable
```

### Step 3: Evidence Retrieval
```python
# For each claim:
1. Generate query embedding
2. Search ChromaDB for top 7 similar passages
3. Retrieve relevant novel excerpts
```

### Step 4: Consistency Analysis
```python
# For each claim:
1. Prompt Gemini with:
   - The claim
   - Retrieved evidence passages
   - Reasoning guidelines (causal, behavioral, contradictory)
2. Get JSON response:
   {
     "consistency": "consistent" or "contradict",
     "confidence": 0.0 to 1.0,
     "reasoning": "explanation",
     "key_evidence": "most relevant passage"
   }
```

### Step 5: Final Decision
```python
# Aggregate claim results:
if (claims with "contradict" AND confidence > 0.65) >= 2:
    final_decision = "contradict"
else:
    final_decision = "consistent"
```

---

## 📈 Results

### Validation Performance
- **Accuracy**: 60% on 5-sample validation set
- **Processing Time**: ~2.5 minutes per case
- **Total Time**: ~2-3 hours for 60 test cases

### Output Format

`results/submission.csv`:
```csv
id,prediction,rationale,book,character
95,contradict,"Backstory claims British spy collaboration, but novel shows French-only connections.",The Count of Monte Cristo,Noirtier
136,consistent,"Island isolation and secret writing align with character's scholarly nature.",The Count of Monte Cristo,Faria
```

---

## 💡 Key Features

✅ Handles long documents (100k+ words)
✅ Semantic search across entire narrative
✅ Multi-claim granular analysis
✅ Causal consistency reasoning
✅ Confidence-weighted decisions
✅ Human-readable explanations

---

## ⚠️ Limitations

- Requires API quota (15 requests/min on free tier)
- Processing time: ~2.5 min per case
- Accuracy depends on retrieval quality
- May miss very subtle narrative constraints

---

## 🔧 Technical Details

### Chunking Strategy
- **Size**: 3000 words per chunk
- **Overlap**: 500 words (maintains context continuity)
- **Why**: Balance between context and granularity

### Retrieval Configuration
- **Top-K**: 7 most similar passages
- **Embedding Model**: text-embedding-004 (1536 dimensions)
- **Similarity**: Cosine similarity in vector space

### LLM Parameters
- **Model**: gemini-2.0-flash-exp
- **Temperature**: Default (balanced creativity/consistency)
- **JSON Mode**: Structured output for reliability

---

## 🚦 Future Improvements

- [ ] Fine-tuned embeddings for literary text
- [ ] Temporal reasoning for event sequences
- [ ] Multi-document cross-referencing
- [ ] Explanation quality scoring
- [ ] Support for more books/longer texts
- [ ] Web interface for easy testing

---

## 📝 Example Use Cases

1. **Creative Writing**: Verify character consistency in drafts
2. **Literary Analysis**: Identify narrative contradictions
3. **Game Development**: Validate character lore
4. **Education**: Teaching narrative consistency
5. **Research**: Automated fact-checking in fiction

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **KDSH 2026 Hackathon** for the challenge
- **Google Gemini Team** for API access
- **ChromaDB** for vector database
- Classic novels: Alexandre Dumas, Jules Verne

---

## 👨‍💻 Author

**Your Name**

- GitHub: https://github.com/Ajinkyawagh4846
- LinkedIn: https://www.linkedin.com/in/ajinkya-wagh-a201212b8/
- Email: ajinkyawagh2005@gmail.com

---

## 📞 Support

Having issues? 
1. Check [Issues](https://github.com/Ajinkyawagh4846/character-backstory-validator/issues)
2. Create a new issue with details
3. Contact via email

---

⭐ **Star this repo if you find it helpful!** ⭐

---

*Built with ❤️ for KDSH 2026 Hackathon*
