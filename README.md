# OCC

## Setup

```
git clone https://github.com/zcahkwn/occ.git
cd occ
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

---

## Run plots

Plot the PMF and its normal approximation of P(union=X) and P(intersection=Y)
```
python scripts/plot_pmf.py
```

Plot the PMF of Jaccard Index
```
python scripts/plot_jaccard.py
```

Plot the bivariate distribution of P(union=X, intersection=Y)  
```
python scripts/plot_bivariate.py
```