# KDSH 2026 - Character Backstory Consistency Checker

## 🎯 Challenge
Determine if character backstories are consistent with events in classic novels.

## 📊 Dataset
- Train: 80 labeled examples (consistent/contradict)
- Test: 60 cases to predict
- Books: The Count of Monte Cristo, In Search of the Castaways

## 🏗️ Approach
- Multi-stage reasoning with Gemini 2.5
- Semantic retrieval with embeddings
- Claim-level verification
- Causal consistency checking

## 🚀 Setup
1. Install Python 3.8+
2. Create virtual environment:
   ```bash
   python -m venv venv
   # Mac/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r code/requirements.txt
   ```
4. Get Gemini API key from `https://aistudio.google.com/app/apikey`
5. Create `.env` file:
   ```bash
   GEMINI_API_KEY=AIzaSyBIJKf3jdLxIbZzmvH_sNDBZoTi1CP61vs
   ```

## 📂 Project Structure
kdsh_2026/
├── books/              # Novel text files  
├── code/               # Python modules  
├── results/            # Output CSV  
├── train.csv           # Training data  
├── test.csv            # Test data  
└── .env                # API key (create this)  

## 🎮 Usage
Run exploration:
```bash
python code/explore_data.py
```

Run main pipeline:
```bash
python code/main.py
```

## 📈 Output
Results saved to: `results/submission.csv`  
Columns: `id`, `prediction`, `rationale`, `book`, `character`

## 🧠 Technology
- Gemini 2.5 Flash for reasoning
- Text Embedding 004 for retrieval
- ChromaDB for vector storage
- Multi-hop causal reasoning

## ⚠️ Notes
- Free tier: 15 req/min for Flash
- Process 60 test cases ~ 10-15 minutes
- Estimated cost: $0 (within free tier)


