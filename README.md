# Elasticsearch Name Search API (Python)

This API provides an endpoint to search for names in Elasticsearch, supporting both exact matches and similar name matches. Built with Python 3.11+ using FastAPI and incorporates intelligent search rules from `cursor_rules.json`.

## Features

- 🔍 **Intelligent Search**: Uses cursor rules to determine search type and relevant fields
- 🎯 **Multi-field Search**: Searches across multiple fields based on query context
- 🌐 **Multi-language Support**: Supports Hebrew and English search patterns
- ⚡ **Fast & Async**: Built with FastAPI for high performance
- 📚 **Auto Documentation**: Interactive API documentation with Swagger UI
- 🎚️ **Configurable Scoring**: Exact matches, fuzzy matching, and partial matches with different boost levels

## Setup

### Prerequisites
- Python 3.11 or higher
- Elasticsearch instance running
- `cursor_rules.json` file in the project root

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```env
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=your_username
ELASTICSEARCH_PASSWORD=your_password
PORT=8000
HOST=0.0.0.0
```

3. Make sure your Elasticsearch instance has an index named `names` with the following mapping:
```json
{
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "role": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "title": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "skills": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "tags": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "team": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      }
    }
  }
}
```

## Usage

### Start the server:

**Option 1: Using the run script**
```bash
python run.py
```

**Option 2: Direct uvicorn**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Using the module directly**
```bash
python -m src.main
```

### API Endpoints

#### 1. Search Endpoint
```
GET /api/search/name?name=<query>
```

**Example Requests:**
```bash
# Basic name search
curl "http://localhost:8000/api/search/name?name=John"

# Technology search (will use technology_tag rules)
curl "http://localhost:8000/api/search/name?name=React"

# Hebrew search (will use person_search rules)
curl "http://localhost:8000/api/search/name?name=מפתח"
```

**Response Format:**
```json
{
  "total": 2,
  "search_type": "person_search",
  "query": "מפתח",
  "results": [
    {
      "name": "יוחנן כהן",
      "score": 5.0,
      "search_type": "person_search",
      "fields_matched": ["role", "title"],
      "exact_match": false,
      "source": {
        "name": "יוחנן כהן",
        "role": "מפתח בכיר",
        "team": "פיתוח"
      },
      "highlights": {
        "role": ["<em>מפתח</em> בכיר"]
      }
    }
  ]
}
```

#### 2. Get Search Rules
```
GET /api/search/rules
```
Returns the loaded cursor rules configuration.

#### 3. Health Check
```
GET /api/health
```
Returns server status and configuration info.

#### 4. Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Search Rule Types

The API automatically detects search intent based on `cursor_rules.json`:

- **person_search**: Searches for people by role/title keywords
- **technology_tag**: Searches for technology skills and tools
- **organizational_affiliation**: Searches within teams/departments
- **service_provider**: Searches for service responsibilities
- **experience**: Searches based on experience levels
- **tags**: General tag-based searches
- **proximity**: Organizational proximity searches
- **visual_features**: Appearance-based searches
- **system_search**: System and tool searches

## Development

### Project Structure
```
├── src/
│   └── main.py          # FastAPI application
├── cursor_rules.json    # Search rules configuration
├── requirements.txt     # Python dependencies
├── run.py              # Development server script
├── .env                # Environment variables
└── README.md           # This file
```

### Adding New Search Rules

Edit `cursor_rules.json` to add new search patterns:

```json
{
  "rules": [
    {
      "type": "custom_search",
      "keywords": ["keyword1", "keyword2"],
      "fields": ["field1", "field2"]
    }
  ]
}
```

The API will automatically reload the rules and apply them to incoming search queries.
