# Manifold & Pipe Comprehensive Audit Plan

## Overview

The Anthropic manifold was developed with deep investigation of OpenWebUI's frontend behavior. Other manifolds were developed without this analysis and may have hidden bugs. This document outlines what to check.

---

## Files to Audit

| File | Type | Status |
|------|------|--------|
| `functions/pipes/anthropic_manifold/anthropic_manifold.py` | Anthropic Claude API | ✅ Audited |
| `suurt8ll_functions/plugins/pipes/gemini_manifold.py` | Gemini API | ⏳ Pending |
| `functions/pipes/gemini_deep_research/gemini_deep_research.py` | Gemini Deep Research | ⏳ Pending |
| `tools/gemini_image_gen/gemini_image_gen.py` | Gemini Image Generation | ⏳ Pending |
| `functions/filters/create_visual_filter/create_visual_filter.py` | Image Generation Filter | ⏳ Pending |

---

## Audit Categories

### 1. Streaming Content Delivery

**Problem:** OpenWebUI splits large `yield` content into 1-3 char chunks, breaking markdown parsing.

**What to check:**
- Any `yield` of code blocks (` ``` `)
- Any `yield` of large formatted text (> 100 chars)
- Any `yield` of HTML blocks

**Fix:** Use `event_emitter({"type": "message", "data": {"content": ...}})` instead of `yield` for atomic delivery.

**Reference:** See `anthropic_manifold.py` lines 1485-1499 for `_emit_message_delta()` helper.

---

### 2. Status Update Management

**Problem:** Status messages can get stuck if not properly cleared.

**What to check:**
- Every `_emit_status(done=False)` should eventually have a corresponding `done=True` or be replaced
- Long-running operations should show progress
- Error paths should clear status

**Example issues:**
- "Generating..." status stuck forever
- "Processing..." never clears after error

---

### 3. Error Handling

**Problem:** Errors may not be properly surfaced to the user.

**What to check:**
- API errors (4xx, 5xx responses)
- Rate limiting (429)
- Timeout handling
- Network failures
- JSON parsing errors
- Invalid response formats

**Best practice:**
```python
try:
    # API call
except Exception as e:
    await event_emitter({
        "type": "chat:completion",
        "data": {"error": {"message": str(e)}, "done": True}
    })
    return
```

---

### 4. HTTP Status Code Handling

**Problem:** REST APIs return different status codes for success.

**What to check:**
- File uploads: Should accept 201 (Created), not just 200
- POST requests: May return 201 or 202
- DELETE requests: May return 204 (No Content)

**Example bug found:** `resp.status == 200` should be `resp.status in (200, 201)` for file uploads.

---

### 5. Message Format Compatibility

**Problem:** Cross-model chats have messages from different formats.

**What to check:**
- How does the pipe handle messages from OTHER models in history?
- Does it gracefully skip unknown content types?
- Does it avoid sending unsupported content to the API?

**Reference:** See `anthropic_manifold.py` `_process_messages()` for defensive handling.

---

### 6. Image Handling

**Problem:** Images may be in various formats (base64, URLs, file references).

**What to check:**
- Base64 data URL parsing (`data:image/png;base64,...`)
- OpenWebUI file references (`/api/v1/files/{id}/content`)
- URL image fetching
- Image format validation (jpeg, png, gif, webp)
- Image size limits

---

### 7. Tool/Function Calling

**Problem:** Tool results may not be properly handled or displayed.

**What to check:**
- Tool call accumulation during streaming
- Tool result parsing
- Display of tool inputs/outputs to user
- Error handling for failed tool calls

---

### 8. API Response Parsing

**Problem:** API responses may have unexpected formats.

**What to check:**
- SSE event parsing (data: prefix handling)
- JSON parsing with error handling
- Handling of empty responses
- Handling of partial/incomplete responses
- Delta vs full content handling

---

### 9. Memory/Resource Management

**Problem:** Resources may not be properly cleaned up.

**What to check:**
- HTTP session reuse vs creation
- Connection pooling
- File handle cleanup
- Memory accumulation in long conversations

---

### 10. Logging & Debugging

**Problem:** Insufficient logging makes debugging difficult.

**What to check:**
- Are errors logged with context?
- Are API requests/responses logged (at debug level)?
- Are there debug toggles in valves?

---

## Gemini Manifold Specific Checks

### Code Execution Display
- **Location:** Lines 3745-3773
- **Issue:** Code blocks are returned as formatted strings - how are they yielded?
- **Check:** Trace call path to see if these go through `yield`

### Grounding/Search Results
- **Check:** How are search results formatted and delivered?
- **Check:** Are citations properly linked?

### Safety Filters
- **Check:** How are blocked responses handled?
- **Check:** Does the user see a clear message?

---

## Gemini Deep Research Specific Checks

### Long-Running Operations
- **Check:** Research can take minutes - is status properly updated?
- **Check:** Can user cancel mid-research?

### Interaction ID Handling
- **Check:** `__INTERACTION_ID__:` prefix - is this properly stripped/handled?

### Output Assembly
- **Check:** How is final research output assembled and delivered?
- **Check:** Are there issues with large reports?

---

## Gemini Image Gen Specific Checks

### Image Storage
- **Check:** Are generated images properly stored in OpenWebUI?
- **Check:** File ID generation and registration

### Error Messages
- **Check:** Safety filter blocks - clear user message?
- **Check:** Rate limit handling

### Multiple Images
- **Check:** How are multiple generated images handled?

---

## Testing Approach

For each manifold, test:

1. **Basic streaming** - Simple prompt, verify text streams correctly
2. **Code blocks** - Ask for code, verify formatting during and after streaming
3. **Long responses** - Generate long content, check for truncation/issues
4. **Errors** - Trigger API errors, verify user sees clear message
5. **Images** - Send/receive images, verify display
6. **Mixed history** - Use in chat with messages from other models
7. **Tool use** - If applicable, test tool/function calling

---

## Reporting Template

For each issue found:

```markdown
## Issue: [Title]

**File:** [path/to/file.py]
**Line:** [line number(s)]
**Severity:** Critical / High / Medium / Low

### Problem
[Description of what's wrong]

### Impact
[What users experience]

### Root Cause
[Technical explanation]

### Fix
[Code changes needed]
```
