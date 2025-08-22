
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

def extract_keyinfo_prompt(query,contents):
    prompt = """# Task
Given several contents and a query, select semantically relevant contents for the query.

# Requirement
1. The selected contents must be able to support answering the given query.
2. You should only return the indices of the contents (without including the content texts).
3. The answer must follow the format:
```json
[number1, number2, ...]
```
4. If no contents are relevant to the query, return an empty list: `[]`

# Examples
- query: Who is the tallest basketball player in China?
- contents:
0. Yao Ming, 2.21m
1. Zhou Qi, 2.16m
2. O'Neal, 2.16m
- response:
```json
[0, 1]
```

# Select related contents for the following query
- query: {query_placeholder}
- contents:
{content_placeholder}
- response:
"""
    text = prompt.format(query_placeholder=query,content_placeholder=contents)
    return text

def validate_contents_prompt(query,contents):
    prompt = """# Task
Given several contents and a query, indetify whether these contents are able to support answering the query.

# Requirement
1. The answer must be either `yes` or `no`.
2. The answer must follow the format:
```json
["..."]
```

# Examples
- query: Who is the tallest basketball player in China?
- contents:
0. Yao Ming, 2.21m
1. Zhou Qi, 2.16m
2. O'Neal, 2.16m
- response:
```json
["yes"]
```

# Judging related contents for the following query
- query: {query_placeholder}
- contents:
{content_placeholder}
- response:
"""
    text = prompt.format(query_placeholder=query,content_placeholder=contents)
    return text

def supplement_query_prompt(query,contents):
    prompt = """`# Task
Given a query and several related contents that are insufficient to answer the query, generate a supplementary query to search for potential answers on the internet.

# Requirement
1. The generated supplementary query should address the information gaps in the current contents to help answer the original query.
2. The answer must strictly follow the format:
```json
["supplementary_query"]
```

# Examples
- query: Who is the tallest basketball player in China?
- contents:
0. Yao Ming
1. Zhou Qi
- response:
```json
["What are the heights of Yao Ming and Zhou Qi?"]
```

# Generate a supplementary query for the following query and contents
- query: {query_placeholder}
- contents:
{content_placeholder}
- response:
"""
    text = prompt.format(query_placeholder=query, content_placeholder=contents)
    return text

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