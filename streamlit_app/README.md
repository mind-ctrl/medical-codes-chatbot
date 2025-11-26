# Medical Coding Assistant - Streamlit Chatbot

Beautiful, interactive chatbot UI for the Medical Coding RAG system.

## Features

- 🎨 **Beautiful UI**: Clean, professional medical app design
- 💬 **Chat-like Interface**: Easy to use for medical professionals
- 🔍 **Three Search Modes**: Quick, Standard, Expert
- 📊 **Visual Results**: Color-coded confidence scores
- 📋 **Example Queries**: Pre-loaded common medical scenarios
- ⚡ **Real-time Stats**: Live database statistics
- 🎯 **Organized Display**: Separate tabs for CPT and ICD-10 codes

## Quick Start

### Prerequisites

1. **Backend must be running** on http://localhost:8000
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Database must be loaded** with CPT and ICD-10 codes

### Installation

```bash
# Navigate to streamlit app directory
cd streamlit_app

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

## Usage

### 1. Check Connection
- Sidebar shows "✅ API Connected" if backend is running
- Shows database statistics (CPT and ICD-10 code counts)

### 2. Enter Query
- Type a clinical description in the text area
- Or click an example query from the sidebar

### 3. Choose Settings
- **Search Mode**:
  - **Quick**: Fast hybrid search (~200ms)
  - **Standard**: Optimized for common queries
  - **Expert**: LLM-powered with explanations (~2s)
- **Results per type**: 1-10 codes

### 4. View Results
- **CPT Codes Tab**: Procedure codes
- **ICD-10 Codes Tab**: Diagnosis codes
- Each code shows:
  - Code number
  - Description
  - Category/Chapter
  - Confidence score (color-coded)
  - Reasoning (Expert mode only)

## Example Queries

Try these in the app:

1. "Patient with type 2 diabetes"
2. "Chest pain with hypertension"
3. "Annual wellness visit"
4. "Acute bronchitis with cough"
5. "Knee replacement surgery"

## Features Explained

### Color-Coded Confidence
- 🟢 **Green** (>80%): High confidence
- 🟡 **Yellow** (60-80%): Medium confidence
- 🔴 **Red** (<60%): Low confidence

### Search Modes
- **Quick**: Hybrid search (vector + keyword) without LLM
- **Standard**: Same as quick (future: cached results)
- **Expert**: Uses Perplexity LLM for reranking and explanations

### Real-time Stats
- Total CPT codes in database
- Total ICD-10 codes in database
- API connection status

## Troubleshooting

### "API Disconnected" Error
**Solution**: Make sure the backend is running
```bash
cd backend
uvicorn app.main:app --reload
```

### "Connection error"
**Causes**:
- Backend not running
- Backend running on different port
- Firewall blocking connection

**Fix**: Check backend is at http://localhost:8000/health

### Slow Response in Expert Mode
**Normal**: Expert mode uses LLM and takes 1.5-2.5 seconds
**Speed up**: Use Quick or Standard mode

## Configuration

Edit `app.py` line 11 to change API URL:
```python
API_BASE_URL = "http://localhost:8000"  # Change if needed
```

## Architecture

```
Streamlit App (Port 8501)
        ↓
   HTTP Requests
        ↓
FastAPI Backend (Port 8000)
        ↓
PostgreSQL + pgvector
        ↓
  Medical Codes
```

## Screenshots

### Main Interface
- Clean, professional design
- Example queries in sidebar
- Real-time search

### Results Display
- Separate tabs for CPT/ICD-10
- Color-coded confidence scores
- Expandable explanations

### Search Modes
- Toggle between Quick/Standard/Expert
- Adjustable result count
- Processing time display

## Tips

1. **Use Example Queries** to get started quickly
2. **Try Expert Mode** for complex cases needing explanation
3. **Quick Mode** is best for fast lookups
4. **Check confidence scores** - higher is better
5. **Read explanations** in Expert mode for learning

## Development

### File Structure
```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Customization

**Change colors**: Edit CSS in `app.py` lines 21-55

**Add features**:
- Code history/favorites
- Export to CSV
- Multiple queries at once
- Category filters

### Dependencies
- `streamlit>=1.28.0` - Web UI framework
- `requests>=2.31.0` - HTTP client for API calls

## Future Enhancements

- [ ] Search history
- [ ] Favorite codes
- [ ] Export results to CSV
- [ ] Batch processing
- [ ] Category filters
- [ ] Code comparison
- [ ] Analytics dashboard

## Support

For issues:
1. Check backend is running
2. Verify database is loaded
3. Check console for errors
4. See main project README

## License

Same as main project (MIT)
