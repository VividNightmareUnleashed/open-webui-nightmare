"""
title: AskUserQuestion
id: AskUserQuestion
version: 0.1.4
description: Prompt the user with a structured form (checkboxes, selects, text) and return the answers to the model.
license: MIT
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class FormField(BaseModel):
    name: str = Field(
        ...,
        description="Stable key used in the returned values object (e.g. 'destination').",
        validation_alias=AliasChoices("name", "key"),
    )
    label: str = Field(
        default="",
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

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        patched = dict(data)

        raw_name = patched.get("name") or patched.get("key")
        if not patched.get("label") and raw_name:
            patched["label"] = str(raw_name).replace("_", " ").strip().title()

        return patched

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        aliases = {
            "radio": "select",
            "dropdown": "select",
            "combo": "select",
            "multi_select": "multiselect",
            "multi-select": "multiselect",
            "multi choice": "multiselect",
            "boolean": "checkbox",
            "bool": "checkbox",
            "string": "text",
            "integer": "number",
            "int": "number",
        }
        return aliases.get(normalized, normalized)

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, dict):
            value = list(value.values())

        if not isinstance(value, list):
            return value

        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
                continue
            if isinstance(item, dict):
                raw = item.get("value") or item.get("label") or item.get("name") or item.get("title")
                if raw is not None:
                    out.append(str(raw))
                else:
                    out.append(json.dumps(item, ensure_ascii=True, separators=(",", ":")))
                continue

            out.append(str(item))

        return out


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

	const __owui_prevOverflow = document.body.style.overflow;
	document.body.style.overflow = 'hidden';

	let __owui_requestCancel = () => {{}};

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
	  className: 'modal fixed top-0 right-0 left-0 bottom-0 bg-black/30 dark:bg-black/60 w-full h-screen max-h-[100dvh] p-3 flex justify-center z-9999 overflow-y-auto overscroll-contain',
	  role: 'dialog',
	  'aria-modal': 'true',
	}}, []);
	
	const __owui_modal = __owui_el('div', {{
	  className: 'm-auto max-w-full w-[42rem] mx-2 shadow-3xl min-h-fit scrollbar-hidden bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm rounded-4xl border border-white dark:border-gray-850',
	  onmousedown: (e) => {{
	    e.stopPropagation();
	  }},
	}}, []);
	
	const __owui_header = __owui_el('div', {{ className: 'flex justify-between dark:text-gray-300 px-5 pt-4 pb-2' }}, [
	  __owui_el('div', {{ className: 'text-lg font-medium self-center' }}, [__owui_title]),
	  __owui_el(
	    'button',
	    {{
	      type: 'button',
	      className: 'self-center text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200',
	      onclick: (e) => {{
	        e.preventDefault();
	        __owui_requestCancel();
	      }},
	    }},
	    [__owui_el('span', {{ className: 'text-xl leading-none' }}, ['×'])]
	  ),
	]);
	
	const __owui_descriptionEl = __owui_description
	  ? __owui_el('div', {{ className: 'px-5 pb-2 text-sm text-gray-500 dark:text-gray-400' }}, [__owui_description])
	  : null;
	
	const __owui_error = __owui_el('div', {{
	  className: 'hidden mx-5 mt-2 px-4 py-2 rounded-lg bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-200 text-sm',
	}}, []);
	
	const __owui_body = __owui_el('div', {{ className: 'flex flex-col md:flex-row w-full px-5 pb-4 md:space-x-4 dark:text-gray-200' }}, []);
	const __owui_bodyInner = __owui_el('div', {{ className: 'flex flex-col w-full sm:flex-row sm:justify-center sm:space-x-6' }}, []);
	
	const __owui_form = __owui_el('form', {{
	  className: 'flex flex-col w-full',
	  onsubmit: (e) => {{
	    e.preventDefault();
	  }},
	}}, []);
	
	const __owui_fieldsWrap = __owui_el('div', {{ className: 'px-1' }}, []);
	const __owui_fieldsEl = __owui_el('div', {{ className: 'flex flex-col gap-1' }}, []);
	__owui_fieldsWrap.appendChild(__owui_fieldsEl);
	__owui_form.appendChild(__owui_fieldsWrap);
	
	const __owui_buttons = __owui_el('div', {{ className: 'flex justify-end pt-3 text-sm font-medium gap-2' }}, []);
	
	const __owui_cancelBtn = __owui_el('button', {{
	  className: 'px-3.5 py-1.5 text-sm font-medium bg-white hover:bg-gray-100 text-black dark:bg-black dark:text-white dark:hover:bg-gray-900 transition rounded-full',
	  type: 'button',
	}}, [__owui_cancelLabel]);
	
	const __owui_submitBtn = __owui_el('button', {{
	  className: 'px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full',
	  type: 'submit',
	}}, [__owui_submitLabel]);
	
	__owui_buttons.appendChild(__owui_cancelBtn);
	__owui_buttons.appendChild(__owui_submitBtn);
	__owui_form.appendChild(__owui_buttons);
	
	__owui_bodyInner.appendChild(__owui_form);
	__owui_body.appendChild(__owui_bodyInner);
	
	__owui_modal.appendChild(__owui_header);
	if (__owui_descriptionEl) __owui_modal.appendChild(__owui_descriptionEl);
	__owui_modal.appendChild(__owui_error);
	__owui_modal.appendChild(__owui_body);
	__owui_overlay.appendChild(__owui_modal);
	document.body.appendChild(__owui_overlay);
	
	__owui_requestCancel = () => {{
	  __owui_cancelBtn.click();
	}};
	
	const __owui_inputsByName = new Map();

	const __owui_inputClass =
	  'w-full rounded-lg py-2 px-4 text-sm dark:text-gray-300 dark:bg-gray-850 outline-hidden border border-gray-100/30 dark:border-gray-850/30';
	const __owui_checkboxClass = 'size-3.5 rounded cursor-pointer border border-gray-200 dark:border-gray-700';
	const __owui_helpClass = 'text-xs text-gray-500 dark:text-gray-400';
	
	function __owui_renderField(field, idx) {{
	  const wrapper = __owui_el('div', {{ className: 'py-0.5 w-full justify-between' }}, []);
	
	  const labelEl = __owui_el('div', {{ className: 'self-center text-xs font-medium' }}, []);
	  labelEl.appendChild(document.createTextNode(field.label || field.name));
	  if (field.required) {{
	    labelEl.appendChild(__owui_el('span', {{ className: 'text-gray-500' }}, [' *required']));
	  }}
	
	  const labelRow = __owui_el('div', {{ className: 'flex w-full justify-between mb-1.5' }}, [labelEl]);
	  wrapper.appendChild(labelRow);
	
	  const inputId = 'input-variable-' + String(idx);
	  let inputEl = null;
	
	  if (field.type === 'textarea') {{
	    inputEl = __owui_el('textarea', {{
	      id: inputId,
	      className: __owui_inputClass + ' min-h-[96px] resize-y',
	      placeholder: field.placeholder || '',
	      rows: '4',
	      autocomplete: 'off',
	      required: field.required ? 'required' : null,
	    }}, []);
	    if (field.default != null) inputEl.value = String(field.default);
	  }} else if (field.type === 'select' || field.type === 'multiselect') {{
	    inputEl = __owui_el('select', {{
	      id: inputId,
	      className: __owui_inputClass,
	      multiple: field.type === 'multiselect' ? 'multiple' : null,
	      size: field.type === 'multiselect' ? String(Math.min(6, Math.max(3, field.options.length || 3))) : null,
	      required: field.required ? 'required' : null,
	    }}, []);
	
	    if (field.type === 'select' && field.placeholder) {{
	      const opt = __owui_el('option', {{ value: '' }}, [field.placeholder]);
	      opt.disabled = true;
	      opt.selected = field.default == null || String(field.default) === '';
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
	      id: inputId,
	      type: 'checkbox',
	      className: __owui_checkboxClass,
	    }}, []);
	    inputEl.checked = Boolean(field.default);
	  }} else {{
	    const allowed = new Set(['text','email','url','date','time','number']);
	    const t = allowed.has(field.type) ? field.type : 'text';
	    inputEl = __owui_el('input', {{
	      id: inputId,
	      type: t,
	      className: __owui_inputClass,
	      placeholder: field.placeholder || '',
	      autocomplete: 'off',
	      required: field.required ? 'required' : null,
	    }}, []);
	    if (t === 'number') {{
	      if (field.min != null && !Number.isNaN(field.min)) inputEl.min = String(field.min);
	      if (field.max != null && !Number.isNaN(field.max)) inputEl.max = String(field.max);
	      if (field.step != null && !Number.isNaN(field.step)) inputEl.step = String(field.step);
	    }}
	    if (field.default != null) inputEl.value = String(field.default);
	  }}
	
	  if (field.type === 'checkbox') {{
	    const checkboxRow = __owui_el('div', {{ className: 'flex items-center space-x-2' }}, []);
	    const checkboxInner = __owui_el('div', {{ className: 'relative flex justify-center items-center gap-2' }}, [
	      inputEl,
	      __owui_el('label', {{ for: inputId, className: 'text-sm' }}, [field.placeholder || field.label || field.name]),
	    ]);
	    checkboxRow.appendChild(checkboxInner);
	    wrapper.appendChild(checkboxRow);
	  }} else {{
	    const row = __owui_el('div', {{ className: 'flex mt-0.5 mb-0.5 space-x-2' }}, [
	      __owui_el('div', {{ className: 'flex-1' }}, [inputEl]),
	    ]);
	    wrapper.appendChild(row);
	  }}
	
	  if (field.description) {{
	    wrapper.appendChild(__owui_el('div', {{ className: __owui_helpClass }}, [field.description]));
	  }}
	
	  __owui_inputsByName.set(field.name, {{ field, inputEl }});
	  return wrapper;
	}}

	for (let i = 0; i < __owui_fields.length; i++) {{
	  __owui_fieldsEl.appendChild(__owui_renderField(__owui_fields[i], i));
	}}

function __owui_showError(message) {{
  __owui_error.textContent = String(message || 'Please check your inputs.');
  __owui_error.classList.remove('hidden');
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
      return 'Required: ' + f.label;
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

    async def AskUserQuestion(
        self,
        schema: dict[str, Any],
        __event_call__=None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        """
        Ask the user questions to gather input, clarify requirements, or get decisions during execution.

        This lets you ask questions with the right input type - offer choices via dropdown,
        collect text responses, get yes/no answers via checkbox, request numbers, dates, etc.

        WHEN TO USE:
        1. Clarify ambiguous instructions with multiple questions
        2. Get decisions requiring structured input or multiple choices
        3. Offer options and let user choose direction
        4. Gather user preferences, settings, or details
        5. Collect information with specific formats (numbers, dates, emails)

        DON'T USE FOR:
        - Single yes/no questions (just ask directly in conversation)
        - Simple single questions (just ask normally)
        - Quick confirmations (ask directly)

        CHOOSING INPUT TYPES:
        - select: Choose ONE from options (like radio buttons) - use for decisions between 2-10 choices
        - multiselect: Choose MULTIPLE from options - use when several can apply
        - checkbox: Single yes/no toggle (within a larger form, not alone)
        - text: Short free-form answer
        - textarea: Longer explanation or description
        - number: Numeric input (can set min/max/step constraints)
        - email/url/date/time: Validated specific formats

        TIPS:
        - Mark questions required only if you truly can't proceed without them
        - Provide clear labels and descriptions
        - Set sensible defaults when you can anticipate the answer
        - Use select/multiselect for constrained choices, text for open-ended
        - Customize submit button text to match context
        - Keep focused (5-8 questions works well, but adjust as needed)

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

    # Backwards-compatible alias for older prompts/configs.
    async def prompt_form(
        self,
        schema: dict[str, Any],
        __event_call__=None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        return await self.AskUserQuestion(
            schema=schema,
            __event_call__=__event_call__,
            __event_emitter__=__event_emitter__,
        )
