"""
title: Form Prompt
id: form_prompt
version: 0.1.1
description: Prompt the user with a structured form (checkboxes, selects, text) and return the answers to the model.
license: MIT
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class FormField(BaseModel):
    name: str = Field(
        ...,
        description="Stable key used in the returned values object (e.g. 'destination').",
    )
    label: str = Field(
        ...,
        description="Human-readable label shown to the user.",
    )
    type: Literal[
        "text",
        "textarea",
        "number",
        "select",
        "multiselect",
        "checkbox",
        "email",
        "url",
        "date",
        "time",
    ] = Field(
        default="text",
        description="Input control type.",
    )

    required: bool = Field(
        default=False,
        description="Whether the user must provide a value.",
    )

    options: list[str] | None = Field(
        default=None,
        description="Options for 'select' or 'multiselect'.",
    )

    placeholder: str | None = Field(
        default=None,
        description="Placeholder text for text-like inputs.",
    )

    description: str | None = Field(
        default=None,
        description="Helper text displayed under the field.",
    )

    default: Any | None = Field(
        default=None,
        description="Default value (type depends on field type).",
    )

    min: float | None = Field(
        default=None,
        description="Min value for 'number'.",
    )
    max: float | None = Field(
        default=None,
        description="Max value for 'number'.",
    )
    step: float | None = Field(
        default=None,
        description="Step for 'number'.",
    )


class FormSchema(BaseModel):
    title: str = Field(
        default="Input required",
        description="Modal title.",
    )
    description: str | None = Field(
        default=None,
        description="Optional description displayed under the title.",
    )
    submit_label: str = Field(
        default="Done",
        description="Submit button label.",
    )
    cancel_label: str = Field(
        default="Cancel",
        description="Cancel button label.",
    )

    fields: list[FormField] = Field(
        ...,
        description="Fields to render.",
        min_length=1,
    )


def _schema_to_b64(schema: dict[str, Any]) -> str:
    payload = json.dumps(schema, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _build_execute_code(schema: FormSchema) -> str:
    schema_b64 = _schema_to_b64(schema.model_dump())

    # This code runs inside Open WebUI's built-in `execute` event handler:
    # Chat.svelte wraps it in an async IIFE and returns the result via Socket.IO ack.
    return f"""
const __owui_schema = JSON.parse(atob("{schema_b64}"));

const __owui_themeIsDark = document.documentElement.classList.contains('dark');
const __owui_prevOverflow = document.body.style.overflow;
document.body.style.overflow = 'hidden';

function __owui_el(tag, attrs, children) {{
  const el = document.createElement(tag);
  if (attrs) {{
    for (const [k, v] of Object.entries(attrs)) {{
      if (v === undefined || v === null) continue;
      if (k === 'style' && typeof v === 'object') {{
        Object.assign(el.style, v);
      }} else if (k.startsWith('on') && typeof v === 'function') {{
        el.addEventListener(k.slice(2).toLowerCase(), v);
      }} else if (k === 'className') {{
        el.className = String(v);
      }} else {{
        el.setAttribute(k, String(v));
      }}
    }}
  }}
  if (children) {{
    for (const child of children) {{
      if (child === undefined || child === null) continue;
      if (typeof child === 'string') {{
        el.appendChild(document.createTextNode(child));
      }} else {{
        el.appendChild(child);
      }}
    }}
  }}
  return el;
}}

function __owui_normalizeField(raw) {{
  const f = (raw && typeof raw === 'object') ? raw : {{}};
  const name = (f.name != null) ? String(f.name) : '';
  const label = (f.label != null) ? String(f.label) : name;
  const type = (f.type != null) ? String(f.type) : 'text';
  const required = Boolean(f.required);
  const options = Array.isArray(f.options) ? f.options.map(String) : [];
  const placeholder = (f.placeholder != null) ? String(f.placeholder) : '';
  const description = (f.description != null) ? String(f.description) : '';
  const def = f.default;
  const min = (f.min != null && f.min !== '') ? Number(f.min) : null;
  const max = (f.max != null && f.max !== '') ? Number(f.max) : null;
  const step = (f.step != null && f.step !== '') ? Number(f.step) : null;
  return {{ name, label, type, required, options, placeholder, description, default: def, min, max, step }};
}}

function __owui_isEmptyValue(fieldType, value) {{
  if (fieldType === 'checkbox') return value !== true;
  if (fieldType === 'multiselect') return !Array.isArray(value) || value.length === 0;
  if (fieldType === 'number') return value === null || Number.isNaN(value);
  if (typeof value === 'string') return value.trim().length === 0;
  return value == null;
}}

function __owui_cleanup(overlay, onKeydown) {{
  try {{
    window.removeEventListener('keydown', onKeydown);
  }} catch {{}}
  try {{
    if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
  }} catch {{}}
  try {{
    document.body.style.overflow = __owui_prevOverflow;
  }} catch {{}}
}}

const __owui_title = (__owui_schema && __owui_schema.title) ? String(__owui_schema.title) : 'Input required';
const __owui_description = (__owui_schema && __owui_schema.description != null) ? String(__owui_schema.description) : '';
const __owui_submitLabel = (__owui_schema && __owui_schema.submit_label) ? String(__owui_schema.submit_label) : 'Done';
const __owui_cancelLabel = (__owui_schema && __owui_schema.cancel_label) ? String(__owui_schema.cancel_label) : 'Cancel';
const __owui_fieldsRaw = Array.isArray(__owui_schema && __owui_schema.fields) ? __owui_schema.fields : [];
const __owui_fields = __owui_fieldsRaw.map(__owui_normalizeField).filter((f) => f.name.length > 0);

if (__owui_fields.length === 0) {{
  document.body.style.overflow = __owui_prevOverflow;
  return {{ error: 'Form schema has no fields.' }};
}}

const __owui_overlay = __owui_el('div', {{
  style: {{
    position: 'fixed',
    inset: '0',
    background: 'rgba(0,0,0,0.6)',
    zIndex: '2147483647',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '16px',
  }},
}}, []);

const __owui_modal = __owui_el('div', {{
  style: {{
    width: '100%',
    maxWidth: '720px',
    maxHeight: '90vh',
    overflow: 'auto',
    borderRadius: '24px',
    padding: '20px 20px 16px 20px',
    background: __owui_themeIsDark ? 'rgba(17,24,39,0.97)' : 'rgba(255,255,255,0.97)',
    color: __owui_themeIsDark ? '#e5e7eb' : '#111827',
    border: __owui_themeIsDark ? '1px solid rgba(55,65,81,0.7)' : '1px solid rgba(229,231,235,0.9)',
    boxShadow: '0 24px 70px rgba(0,0,0,0.35)',
    fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, Apple Color Emoji, Segoe UI Emoji',
  }},
}}, []);

const __owui_header = __owui_el('div', {{
  style: {{
    marginBottom: '12px',
  }}
}}, [
  __owui_el('div', {{ style: {{ fontSize: '18px', fontWeight: '600', marginBottom: __owui_description ? '6px' : '0' }} }}, [__owui_title]),
  __owui_description ? __owui_el('div', {{ style: {{ fontSize: '13px', lineHeight: '1.4', opacity: '0.85' }} }}, [__owui_description]) : null,
]);

const __owui_error = __owui_el('div', {{
  style: {{
    display: 'none',
    margin: '10px 0 0 0',
    padding: '10px 12px',
    borderRadius: '12px',
    background: __owui_themeIsDark ? 'rgba(127,29,29,0.5)' : 'rgba(254,226,226,1)',
    color: __owui_themeIsDark ? '#fecaca' : '#7f1d1d',
    fontSize: '13px',
  }}
}}, []);

const __owui_form = __owui_el('form', {{
  style: {{
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    marginTop: '14px',
  }},
  onsubmit: (e) => {{
    e.preventDefault();
  }},
}}, []);

const __owui_inputsByName = new Map();

function __owui_inputBaseStyle() {{
  return {{
    width: '100%',
    boxSizing: 'border-box',
    padding: '10px 12px',
    borderRadius: '12px',
    border: __owui_themeIsDark ? '1px solid rgba(55,65,81,0.9)' : '1px solid rgba(229,231,235,1)',
    background: __owui_themeIsDark ? 'rgba(31,41,55,0.7)' : '#ffffff',
    color: __owui_themeIsDark ? '#f3f4f6' : '#111827',
    outline: 'none',
    fontSize: '14px',
  }};
}}

function __owui_labelStyle() {{
  return {{
    fontSize: '12px',
    fontWeight: '600',
    marginBottom: '6px',
  }};
}}

function __owui_helpStyle() {{
  return {{
    fontSize: '12px',
    opacity: '0.8',
    marginTop: '6px',
    lineHeight: '1.35',
  }};
}}

function __owui_renderField(field) {{
  const wrapper = __owui_el('div', null, []);
  const labelRow = __owui_el('div', {{ style: __owui_labelStyle() }}, [
    field.label + (field.required ? ' *' : ''),
  ]);

  let inputEl = null;
  if (field.type === 'textarea') {{
    inputEl = __owui_el('textarea', {{
      placeholder: field.placeholder || '',
      style: Object.assign(__owui_inputBaseStyle(), {{ minHeight: '96px', resize: 'vertical' }}),
      rows: '4',
    }}, []);
    if (field.default != null) inputEl.value = String(field.default);
  }} else if (field.type === 'select' || field.type === 'multiselect') {{
    inputEl = __owui_el('select', {{
      style: __owui_inputBaseStyle(),
      multiple: field.type === 'multiselect' ? 'multiple' : null,
      size: field.type === 'multiselect' ? String(Math.min(6, Math.max(3, field.options.length || 3))) : null,
    }}, []);
    if (field.type === 'select' && field.placeholder) {{
      const opt = __owui_el('option', {{ value: '' }}, [field.placeholder]);
      opt.disabled = field.required ? true : false;
      opt.hidden = field.required ? true : false;
      inputEl.appendChild(opt);
    }}
    for (const opt of field.options) {{
      inputEl.appendChild(__owui_el('option', {{ value: opt }}, [opt]));
    }}

    const def = field.default;
    if (field.type === 'multiselect' && Array.isArray(def)) {{
      for (const optionEl of Array.from(inputEl.options)) {{
        optionEl.selected = def.map(String).includes(optionEl.value);
      }}
    }} else if (field.type === 'select' && def != null) {{
      inputEl.value = String(def);
    }}
  }} else if (field.type === 'checkbox') {{
    inputEl = __owui_el('input', {{
      type: 'checkbox',
      style: {{
        width: '16px',
        height: '16px',
      }},
    }}, []);
    inputEl.checked = Boolean(field.default);
  }} else {{
    const allowed = new Set(['text','email','url','date','time','number']);
    const t = allowed.has(field.type) ? field.type : 'text';
    inputEl = __owui_el('input', {{
      type: t,
      placeholder: field.placeholder || '',
      style: __owui_inputBaseStyle(),
    }}, []);
    if (t === 'number') {{
      if (field.min != null && !Number.isNaN(field.min)) inputEl.min = String(field.min);
      if (field.max != null && !Number.isNaN(field.max)) inputEl.max = String(field.max);
      if (field.step != null && !Number.isNaN(field.step)) inputEl.step = String(field.step);
    }}
    if (field.default != null) inputEl.value = String(field.default);
  }}

  if (field.type === 'checkbox') {{
    const row = __owui_el('div', {{ style: {{ display: 'flex', alignItems: 'center', gap: '10px' }} }}, [
      inputEl,
      __owui_el('div', {{ style: {{ fontSize: '14px', fontWeight: '500', opacity: '0.95' }} }}, [field.placeholder || '']),
    ]);
    wrapper.appendChild(labelRow);
    wrapper.appendChild(row);
  }} else {{
    wrapper.appendChild(labelRow);
    wrapper.appendChild(inputEl);
  }}

  if (field.description) {{
    wrapper.appendChild(__owui_el('div', {{ style: __owui_helpStyle() }}, [field.description]));
  }}

  __owui_inputsByName.set(field.name, {{ field, inputEl }});
  return wrapper;
}}

for (const f of __owui_fields) {{
  __owui_form.appendChild(__owui_renderField(f));
}}

const __owui_buttons = __owui_el('div', {{
  style: {{
    display: 'flex',
    gap: '10px',
    marginTop: '16px',
  }},
}}, []);

function __owui_buttonStyle(kind) {{
  const base = {{
    flex: '1',
    padding: '10px 12px',
    borderRadius: '999px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '600',
  }};
  if (kind === 'primary') {{
    return Object.assign(base, {{
      background: __owui_themeIsDark ? '#f3f4f6' : '#111827',
      color: __owui_themeIsDark ? '#111827' : '#f9fafb',
    }});
  }}
  return Object.assign(base, {{
    background: __owui_themeIsDark ? 'rgba(31,41,55,0.9)' : 'rgba(243,244,246,1)',
    color: __owui_themeIsDark ? '#f9fafb' : '#111827',
  }});
}}

const __owui_cancelBtn = __owui_el('button', {{
  type: 'button',
  style: __owui_buttonStyle('secondary'),
}}, [__owui_cancelLabel]);

const __owui_submitBtn = __owui_el('button', {{
  type: 'submit',
  style: __owui_buttonStyle('primary'),
}}, [__owui_submitLabel]);

__owui_buttons.appendChild(__owui_cancelBtn);
__owui_buttons.appendChild(__owui_submitBtn);

__owui_modal.appendChild(__owui_header);
__owui_modal.appendChild(__owui_error);
__owui_modal.appendChild(__owui_form);
__owui_modal.appendChild(__owui_buttons);
__owui_overlay.appendChild(__owui_modal);
document.body.appendChild(__owui_overlay);

function __owui_showError(message) {{
  __owui_error.textContent = String(message || 'Please check your inputs.');
  __owui_error.style.display = 'block';
  __owui_modal.scrollTo({{ top: 0, behavior: 'smooth' }});
}}

function __owui_collectValues() {{
  const values = {{}};
  for (const [name, entry] of __owui_inputsByName.entries()) {{
    const field = entry.field;
    const inputEl = entry.inputEl;
    let value = null;

    if (field.type === 'checkbox') {{
      value = Boolean(inputEl.checked);
    }} else if (field.type === 'multiselect') {{
      value = Array.from(inputEl.selectedOptions || []).map((o) => o.value);
    }} else if (field.type === 'number') {{
      value = (inputEl.value === '') ? null : Number(inputEl.value);
    }} else {{
      value = (inputEl.value != null) ? String(inputEl.value) : '';
    }}

    values[name] = value;
  }}
  return values;
}}

function __owui_validate(values) {{
  for (const f of __owui_fields) {{
    if (!f.required) continue;
    if (__owui_isEmptyValue(f.type, values[f.name])) {{
      return `Required: ${{f.label}}`;
    }}
  }}
  return null;
}}

function __owui_focusFirst() {{
  for (const f of __owui_fields) {{
    const entry = __owui_inputsByName.get(f.name);
    if (!entry) continue;
    const el = entry.inputEl;
    if (el && typeof el.focus === 'function') {{
      el.focus();
      return;
    }}
  }}
}}

const __owui_onKeydown = (e) => {{
  if (e.key === 'Escape') {{
    e.preventDefault();
    __owui_cancelBtn.click();
  }}
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {{
    e.preventDefault();
    __owui_submitBtn.click();
  }}
}};

window.addEventListener('keydown', __owui_onKeydown);

__owui_overlay.addEventListener('mousedown', (e) => {{
  if (e.target === __owui_overlay) {{
    __owui_cancelBtn.click();
  }}
}});

__owui_focusFirst();

return await new Promise((resolve) => {{
  let settled = false;

  function finish(result) {{
    if (settled) return;
    settled = true;
    __owui_cleanup(__owui_overlay, __owui_onKeydown);
    resolve(result);
  }}

  __owui_cancelBtn.addEventListener('click', () => {{
    finish({{ cancelled: true }});
  }});

  __owui_form.addEventListener('submit', (e) => {{
    e.preventDefault();
    const values = __owui_collectValues();
    const err = __owui_validate(values);
    if (err) {{
      __owui_showError(err);
      return;
    }}
    finish({{ cancelled: false, values }});
  }});

  __owui_submitBtn.addEventListener('click', (e) => {{
    e.preventDefault();
    __owui_form.requestSubmit();
  }});
}});
""".strip()


class Tools:
    """Tool that prompts the user with a structured form and returns the answers."""

    async def prompt_form(
        self,
        schema: dict[str, Any],
        __event_call__=None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        """
        Show a structured form in the user's browser and return the filled values.

        This uses Open WebUI's built-in `execute` event type (via `__event_call__`)
        to render a modal form (checkboxes, selects, text inputs) and await a user
        response.

        :param schema: Form schema object with keys: title, description, submit_label, cancel_label, fields.
        :return: A dict like {"cancelled": false, "values": {...}} or {"cancelled": true} or {"error": "..."}.
        """
        if __event_call__ is None:
            return {
                "error": "Missing __event_call__. This tool requires WebSocket event calls to the browser.",
            }

        try:
            parsed_schema = FormSchema.model_validate(schema)
        except Exception as exc:
            return {"error": f"Invalid schema: {exc}"}

        if __event_emitter__ is not None:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Waiting for form input…",
                        "done": False,
                    },
                }
            )

        try:
            try:
                code = _build_execute_code(parsed_schema)
                result = await __event_call__(
                    {
                        "type": "execute",
                        "data": {
                            # Some Open WebUI versions/docs use `script`; others use `code`.
                            # Sending both improves compatibility.
                            "code": code,
                            "script": code,
                        },
                    }
                )
            except Exception as exc:
                message = str(exc).strip() or repr(exc) or exc.__class__.__name__
                return {
                    "cancelled": True,
                    "error": f"Form UI failed: {message}",
                }
        finally:
            if __event_emitter__ is not None:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Form complete.",
                            "done": True,
                        },
                    }
                )

        if isinstance(result, dict):
            return result
        return {"value": result}
