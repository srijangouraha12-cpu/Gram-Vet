# GramVet TRUST Build — Integrated ML + Case Management

This build bundles the trained bovine ML models directly inside the Flask application. No separate disease-model server is required.

## What is included
- Farmer report flow with live weather inputs, AI-assisted disease screening, triage risk and escalation risk.
- Permanent animal history: every report, case, treatment, medication note and vaccination remains visible.
- Automatic open case creation and veterinarian notification when a farmer submits a report.
- Vet workflow with TEST, MEDICATION, NOTE and CURE actions.
- A case can only be CLOSED after the assigned veterinarian records a CURE action.
- Structured vaccination recording tied to the animal, case and veterinarian.
- Government vaccine inventory: every recorded vaccination deducts stock and creates an audit transaction.
- Government dashboard with total vaccination events, verified animal coverage, vaccination counts by vaccine and vet, and vaccine-stock usage.
- Animal history screen showing previous cases, times cured and vaccination history.

## Run on Windows
1. Open this folder in VS Code / terminal.
2. Run `run_windows.bat`.
3. The script creates `.venv`, installs `requirements.txt`, and starts `app.py`.
4. Open `http://127.0.0.1:5000`.

## Demo accounts
- Government: `+919000000001` / `Gov@26128`
- Vet One: `+919000000002` / `Vet@26128`
- Vet Two: `+919000000003` / `Vet@26128`
- Vet Three: `+919000000004` / `Vet@26128`
- Demo farmers: `+919000000005` through `+919000000014` / `Farmer@26128`

## ML package
`ml_package/` contains the trained Random Forest disease classifier, Decision Tree triage model, Logistic Regression escalation model, preprocessing artifacts and the dataset used to train them.

The prediction models themselves were not replaced with a new architecture; the Flask app now adapts the farmer/animal information to the bundled model's expected 35-feature input format.
