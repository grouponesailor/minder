const express = require('express');
const { Client } = require('@elastic/elasticsearch');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(express.json());

// Load cursor rules
let cursorRules = {};
try {
  const rulesData = fs.readFileSync(path.join(__dirname, '../cursor_rules.json'), 'utf8');
  cursorRules = JSON.parse(rulesData);
} catch (error) {
  console.error('Error loading cursor rules:', error);
}

// Create Elasticsearch client
const client = new Client({
  node: process.env.ELASTICSEARCH_URL || 'http://localhost:9200',
  auth: {
    username: process.env.ELASTICSEARCH_USERNAME,
    password: process.env.ELASTICSEARCH_PASSWORD
  }
});

// Helper function to determine search type based on query
function determineSearchType(query) {
  const queryLower = query.toLowerCase();
  
  for (const rule of cursorRules.rules || []) {
    // Check keywords
    if (rule.keywords) {
      for (const keyword of rule.keywords) {
        if (queryLower.includes(keyword.toLowerCase())) {
          return rule;
        }
      }
    }
    
    // Check patterns (basic pattern matching)
    if (rule.patterns) {
      for (const pattern of rule.patterns) {
        const patternRegex = pattern.replace(/\*/g, '.*');
        if (new RegExp(patternRegex, 'i').test(query)) {
          return rule;
        }
      }
    }
  }
  
  return { type: 'general', fields: ['name'] };
}

// Build Elasticsearch query based on search type and rules
function buildElasticsearchQuery(query, searchRule) {
  const baseQuery = {
    bool: {
      should: []
    }
  };

  // Add queries for each relevant field
  const fieldsToSearch = searchRule.fields || ['name'];
  
  fieldsToSearch.forEach(field => {
    // Exact match with high boost
    baseQuery.bool.should.push({
      term: {
        [`${field}.keyword`]: {
          value: query,
          boost: 5.0
        }
      }
    });
    
    // Fuzzy match
    baseQuery.bool.should.push({
      match: {
        [field]: {
          query: query,
          fuzziness: "AUTO",
          operator: "and",
          boost: 2.0
        }
      }
    });
    
    // Partial match
    baseQuery.bool.should.push({
      wildcard: {
        [`${field}.keyword`]: {
          value: `*${query}*`,
          boost: 1.5
        }
      }
    });
  });

  return baseQuery;
}

// Enhanced search endpoint
app.get('/api/search/name', async (req, res) => {
  try {
    const { name } = req.query;
    
    if (!name) {
      return res.status(400).json({ error: 'Name parameter is required' });
    }

    // Determine search type based on cursor rules
    const searchRule = determineSearchType(name);
    
    // Build the Elasticsearch query
    const query = buildElasticsearchQuery(name, searchRule);

    const response = await client.search({
      index: 'names', // Make sure this index exists in your Elasticsearch
      body: {
        size: 10,
        query: query,
        highlight: {
          fields: {
            "*": {}
          }
        }
      }
    });

    const hits = response.hits.hits.map(hit => ({
      name: hit._source.name,
      score: hit._score,
      search_type: searchRule.type,
      fields_matched: searchRule.fields,
      exact_match: hit._source.name && hit._source.name.toLowerCase() === name.toLowerCase(),
      source: hit._source,
      highlights: hit.highlight || {}
    }));

    res.json({
      total: response.hits.total.value,
      search_type: searchRule.type,
      query: name,
      results: hits
    });

  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Error performing search', details: error.message });
  }
});

// New endpoint to get available search rules
app.get('/api/search/rules', (req, res) => {
  res.json(cursorRules);
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    rules_loaded: !!cursorRules.rules
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`Loaded ${cursorRules.rules?.length || 0} search rules`);
}); 