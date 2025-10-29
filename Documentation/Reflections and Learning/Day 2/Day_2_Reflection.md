# 🧩 Day 2 Reflection --- Environment Foundations & Git Discipline

## 1️⃣ Importing and Dependency Logic

Today clarified that importing in Python isn't about dragging entire
environments into code --- it's about calling on specific *packages*
that already live inside the active virtual environment.\
When the venv is activated, Python automatically knows where to look
(`Lib/site-packages`), so import statements like `import torch` or
`from lightglue import LightGlue` simply access what's already
installed.\
I learned that importing is the bridge between *installed resources* and
*runtime logic*, not an extra setup step.

------------------------------------------------------------------------

## 2️⃣ Setting Interpreter Paths

The interpreter is what actually executes my Python code, and VS Code
(or any IDE) needs to be told which one to use.\
By selecting my venv's interpreter

    ...Tufts-NASA-Intership\NasaVEnv\Scripts\python.exe

I made sure that both my terminal **and** my editor reference the same
runtime.\
This fixed the "Import cannot be resolved" messages from Pylance and
aligned my development workflow.\
The interpreter path is basically the *voice* my editor uses to speak
Python --- choose the wrong one and it speaks to the wrong environment.

------------------------------------------------------------------------

## 3️⃣ Virtual Environments

A virtual environment isn't a file to import or move --- it's a
*context*.\
Activating it with

    & ".\NasaVEnv\Scripts\Activate.ps1"

redirects all Python and pip commands so they use the venv's internal
interpreter and packages.\
This isolation keeps dependencies clean and prevents conflicts with
global installations.\
In short: **activate, install, import --- not move, copy, or merge.**

------------------------------------------------------------------------

## 4️⃣ GitHub Etiquette & Best Practices

GitHub isn't meant to store massive runtime folders like virtual
environments.\
My earlier attempt to push `NasaVEnv` triggered a "file too large" error
because PyTorch's `.dll` files exceed 100 MB.\
The proper approach is to: - Keep the venv local (never commit it). -
Add `NasaVEnv/` to `.gitignore`. - Track dependencies with\
`pip freeze > requirements.txt` This lets others recreate the same setup
without carrying heavy binaries.\
In essence: **commit your recipe, not your kitchen.**

------------------------------------------------------------------------

## 💡 Summary Insight

Today I built the foundation every serious Python engineer needs:
environment control, interpreter awareness, import literacy, and
source-control discipline.\
From here forward, I can confidently reproduce my setup anywhere --- and
that means I'm ready to focus on *using* LightGlue, not just installing
it.
