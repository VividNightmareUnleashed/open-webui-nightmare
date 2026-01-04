# Regenerate with Pro (Action)

Adds a message action button under assistant messages that regenerates the selected message using the Pro model (`openai_responses.gpt-5.2-pro` by default).

## Install (Open WebUI)

1. Admin Panel → Functions → **New Function**.
2. Type: **Action**
3. Paste `pro_regenerate_action.py` content.
4. Save and enable it.

## Enable the button under messages

Actions show under a message based on the message’s **model**. Enable this action for the model(s) you want:

- Workspace → Models → Edit your model → **Actions** → select **Regenerate with Pro**

Or mark it as a global action in the admin UI.

## Notes

- This regenerates **in-place** (overwrites the existing assistant message content).
- If `pro_filter` is installed, it will use `pro_filter`’s configured `MODEL` valve by default.

