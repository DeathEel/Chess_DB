# Chess_DB
This project consists of two Python scripts designed to process chess games stored in `.pgn` files. The project interacts with the newly created chess database to analyze player performance, game results, and opening strategies. It allows users to execute various queries and generate visualizations such as scatter plots and Elo predictions.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- **pgn_to_csv.py**: Converts chess files stored as `.pgn` to `.csv` for later processing.
- **chess.py**: Performs analyses and generates visualizations on dataset provided.

## Requirements
- Python 3.12
- **Third-party libraries**:
  - `pymysql`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `mlxtend`
  - `huggingface_hub`
  - `dotenv`
- **Standard Python libraries** (no installation required):
  - `re`
  - `datetime`
  - `os`
  - `csv`

## Installation
1. Clone the repository:
```bash
git clone https://github.com/DeathEel/Chess_DB.git
cd Chess_DB
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
**MySQL Database**:
1. Ensure that you have a running MySQL server.
2. Create a database called `chess`.

**Hugging Face user access token**:
1. Create a Hugging Face account from [Hugging Face](https://huggingface.co/join). 
2. Once logged in, click on the profile at the top right corner. 
3. Click on “Access token”, then click on “Create new token”. 
4. Type a token name, check the option “Make calls to the serverless Inference API”. 
5. Scroll to the bottom, click on “Create token”.
6. Store the token as an environment variable. You will not be able to access this value again.

**Environment variables**:
1. Copy the `.env.example` file to `.env`.
```bash
# On Windows:
copy .env.example .env

# On macOS/Linux:
cp .env.example .env
```
2. Open the `.env` file and replace the placeholders with your actual values:
- `HF_TOKEN`: Your Hugging Face API token.
- `DB_USER`: The username for your database connection.
- `DB_PASSWD`: The password for your database.
- `DB_PORT`: The port on which your database is running.
- `CHESS_NAME`: The name associated with the `.pgn` files that is to be analyzed.

## Usage
1. Download the `.pgn` files from chess games and move them to the `input` directory.

2. Run `pgn_to_csv.py` to convert `.pgn` files into a `.csv` file and populate the database:
```bash
python pgn_to_csv.py
```

3. Run `chess.py` to perform analyses or generate visualizations:
```bash
python chess.py
```

4. Look in the `output` directory to see any generated datasets, visualizations, or logs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This project was adapted from a project used for the EECS 118 course at the University of California, Irvine.