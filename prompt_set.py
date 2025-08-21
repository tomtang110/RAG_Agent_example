
def router_prompt(query):
    prompt = """# Background
Users will propose a basketball-related question. There are two basketball-related databases including Chinese and American basketball databases.

# Task
You should identify which database is corresponding to the given query

# Requirement
1. If the query is related only to American basketball (e.g., involving American players, leagues, events), the result is ["america"]
2. If the query is related only to Chinese basketball (e.g., involving Chinese players, leagues, events), the result is ["china"]
3. If the query involves both American and Chinese basketball (e.g., comparing players from both countries) or is a general basketball question not specific to either country (e.g., rules of basketball), the result is ["america","china"]
The response must strictly follow the format:```json
[...]
```

# Examples
- query: who is highest basketball player in China?
- response:```json
["china"]
```

# routing the following query
- query: {query_placeholder}
- response:
"""
    promt = prompt.format(query_placeholder=query)
    return promt

def generation_prompt(query,contents):
    prompt = """# Background
You are provided with retrieval contents related to a query. Your task is to generate a comprehensive and accurate answer to the query using only the information from the retrieval contents. Do not use any external knowledge beyond the given retrieval materials.

# Retrieval Content
{content_placeholder}

# Query
{query_placeholder}

# Requirements
1. The answer must be based strictly on the retrieval content; no additional information is allowed.
2. If the retrieval content contains conflicting information, prioritize the most detailed or recently mentioned (if timestamps exist) data.
3. If the retrieval content does not fully address the query, clearly state the parts that cannot be answered based on the provided information.
4. Organize the answer in a logical and coherent manner, using appropriate formatting (e.g., lists, paragraphs) for readability.
"""
    text = prompt.format(content_placeholder=contents,query_placeholder=query)
    return text