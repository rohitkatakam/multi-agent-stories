# Setup Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the application:**
   ```bash
   python3 main.py
   ```

## Requirements

- Python 3.8+
- OpenAI API key with GPT-3.5-turbo access

## Testing

Test system communication without API calls:
```bash
python3 story_system.py
```

## Troubleshooting

**Missing OpenAI module:**
```bash
pip install openai>=1.0.0
```

**API key not found:**
```bash
echo $OPENAI_API_KEY  # Should display your key
```  