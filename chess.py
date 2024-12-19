import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re
from datetime import datetime, date
import os

initial_run = 1

# Load environment variables from .env files
load_dotenv()

hf_token = os.getenv("HF_ACCESS_TOKEN")
db_user = os.getenv("DB_USER")
db_passwd = os.getenv("DB_PASSWD")
db_port = int(os.getenv("DB_PORT"))
chess_name = os.getenv("CHESS_NAME")

db = pymysql.connect(host='localhost',
                     user=db_user,
                     passwd=db_passwd,
                     db='chess',
                     port=db_port)
cur = db.cursor()

def main():
    selected_query = 0

    while (selected_query != 11):
        selected_query = print_menu()
        execute_query(selected_query)

def execute_query(selected_query):
    match(selected_query):
        case 1:
            date = str(input("[INPUT] Please input a date in \"YYYY.MM.DD\" format: "))
            get_games_played(date)
        case 2:
            opening = str(input("[INPUT] Please input an opening in PGN format (e.g. \"1. e4 e5 2. Nf3 Nc6\"): "))
            get_win_rate_opening(opening)
        case 3:
            try:
                games_played = int(input("[INPUT] Please input the minimum number of games played to filter uncommon openings: "))
            except Exception as e:
                respond(str(e), 2)
                return

            get_top_openings(games_played)
        case 4:
            get_win_loss_draw()
        case 5:
            get_peak_elo()
        case 6:
            respond(f"The scatter plot is being shown. It is also saved as \"output/plots/scatter_elo_result.png\".", 0)
            respond(f"The colors represent the results of the game:", 1)
            respond(f"Forest Green:\t1-0\t\t(White win)", 1)
            respond(f"Orange Red:\t\t0-1\t\t(Black win)", 1)
            respond(f"Indigo:\t\t1/2-1/2\t\t(Draw)", 1)
            respond(f"To continue, close the plot window.", 0)
            generate_scatter_plot()
        case 7:
            date = str(input("[INPUT] Please input a date in \"YYYY.MM.DD\" format: "))
            respond(f"The date plot is being shown. It is also saved as \"output/plots/linear_regression_predict_elo.png\".", 0)
            respond(f"The line represents the linear regression line. It is being used to predict the elo at the inputted date.", 1)
            respond(f"To continue, close the plot window.", 0)
            predict_elo(date)
        case 8:
            find_openings()
        case 9:
            try:
                elo_white = str(input("[INPUT] Please input the Elo of the White player (e.g. \"1000\"): "))
                elo_black = str(input("[INPUT] Please input the Elo of the Black player (e.g. \"1000\"): "))
                respond(f"NOTE: a lower step size means a longer loading time.", 0) 
                respond(f"For a step size of 1, which means a point is plotted for every 1 Elo, the graph takes about 2-3 minutes to load.", 1)
                respond(f"A step size between 1 and 10 is recommended, with 1 for accuracy and 10 for speed.", 1)
                step_size = int(input("[INPUT] Please input the step size for the graph (e.g. \"10\"): "))
            except Exception as e:
                respond(str(e), 2)
                return

            predict_result(elo_white, elo_black, step_size)
        case 10:
            respond(f"To ask questions about the user, use the username \"{chess_name}\" as needed.", 1)
            user_prompt = str(input("[INPUT] Please provide the prompt for the LLM to generate a SQL query: "))
            llm_query(user_prompt)
        case 11:
            respond(f"Thank you for using Chess DB!", 0)

def get_games_played(date):
    row = query(f"SELECT COUNT(*) FROM games WHERE Date = '{date}';")
    respond(f"{row[0][0]} games were played on {date}.", 0)

def get_win_rate_opening(opening):
    rows = query(f"SELECT White, Result, PGN FROM games WHERE PGN LIKE '{opening}%';")
    
    if len(rows) == 0:
        respond(f"There are no games with the opening", 0)
        respond(f"{opening}", 1)

        return

    win_rate = calculate_win_rate(rows)

    respond(f"My win rate with", 0)
    respond(f"{opening}", 1)
    respond(f"is {win_rate:.2f}%.", 1)

def calculate_win_rate(rows):
    wins = 0
    for row in rows:
        white = row[0]
        result = row[1]
        
        if white == chess_name:
            if result == "1-0":
                wins += 1
        elif result == "0-1":
            wins += 1

    return wins / len(rows) * 100

def get_top_openings(games_played):
    rows = query(f"SELECT PGN FROM games;")
    best_openings = []

    try:
        add_opening_to_table()

        # Generate list of all distinct openings
        rows = query(f"SELECT COUNT(*), Opening FROM games GROUP BY Opening;")

        # For each distinct opening, calculate win rate
        openings = []
        for opening in rows:
            # Filter uncommon openings
            if opening[0] <= games_played:
                continue

            # Calculate win rate
            opening_games = query(f"SELECT White, Result, PGN FROM games WHERE Opening = '{opening[1]}';")
            win_rate = calculate_win_rate(opening_games)

            # Append info to list of all openings
            openings.append([opening[1], win_rate, opening[0]])

        # Using list of all openings, find best
        best_openings = find_best_openings(openings)
        remove_opening_from_table()

    except pymysql.err.OperationalError as e:
        if (e.args[0] == 1060):
            respond(str(e.args[0]) + ": " + e.args[1] + f". Column Opening has been dropped. Try again.", 2)
            query(f"ALTER TABLE games DROP COLUMN Opening;")
        else:
            respond(str(e), 2)
            return

    respond(f"The best openings by win rate are", 0)
    for opening in best_openings:
        respond(f"{opening[0]}", 1)
        respond(f"with a win rate of {opening[1]:.2f}% and {opening[2]} games played.\n", 1)

def add_opening_to_table():
    # Temporarily add Opening column for PGN format opening moves
    query(f"ALTER TABLE games ADD Opening varchar(255);")
    rows = query(f"SELECT PGN from games;")

    for opening_list in rows:
        opening_pgn = get_opening(opening_list[0])  # First three moves per player
        query(f"UPDATE games SET Opening = '{opening_pgn}' WHERE PGN = '{opening_list[0].replace("'", "\\'")}';")

def remove_opening_from_table():
    # Remove temporary Opening column
    query(f"ALTER TABLE games DROP COLUMN Opening;")

def get_opening(opening_list):
    pattern = r"(^1\. .*?)(?: \b4\.|$)+?"   # In English: Capture everything from "1." to "4." if "4." exists
    opening = re.findall(pattern, opening_list)[0]

    return opening

def find_best_openings(all_openings):
    # Sort openings by win rate
    all_openings.sort(reverse=True, key=lambda x: x[1])

    # Append first five openings
    best_openings = []
    for i in range(0, 5):
        best_openings.append(all_openings[i])

    return best_openings

def get_win_loss_draw():
    rows = query(f"SELECT White, Result FROM games;")

    wins = 0
    losses = 0
    draws = 0
    for game in rows:
        white = game[0]
        result = game[1]

        if result == "1/2-1/2":
            draws += 1
        elif white == chess_name:
            if result == "1-0":
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            wins += 1
        else:
            losses += 1

    respond(f"I have {wins} wins, {losses} losses, and {draws} draws.", 0)

def get_peak_elo():
    peak_elo = query(f"SELECT MAX(WhiteElo) FROM games WHERE White = '{chess_name}';")[0][0]
    peak_elo = max(query(f"SELECT MAX(BlackElo) FROM games WHERE Black = '{chess_name}';")[0][0], peak_elo)

    respond(f"My max Elo is {peak_elo}.", 0)

def generate_scatter_plot(show_fig=True):
    chess_data = pd.read_csv("output/csv/chess.csv")

    # Color code results
    results_colors = {"1-0": "forestgreen", "0-1": "orangered", "1/2-1/2": "indigo"}
    chess_data["Color"] = chess_data["Result"].map(results_colors)

    plt.scatter(chess_data["WhiteElo"], chess_data["BlackElo"], s=8, c=chess_data["Color"])

    # Create legend
    handles = [plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=8) for color in results_colors.values()]
    labels = ["1-0", "0-1", "1/2-1/2"]
    plt.legend(handles, labels, title="Results", loc="best")

    # Fix labels, title, and axes
    plt.xlabel("White Elo")
    plt.ylabel("Black Elo")
    plt.title("White Elo vs. Black Elo with Result")
    plt.xlim(0, max(chess_data["WhiteElo"].max(), chess_data["BlackElo"].max()) * 1.1)
    plt.ylim(0, max(chess_data["WhiteElo"].max(), chess_data["BlackElo"].max()) * 1.1)
    if show_fig:
        plt.savefig("output/plots/scatter_elo_result.png")
        plt.show()
        plt.clf()

def predict_elo(prediction_date):
    chess_data = pd.read_csv("output/csv/chess.csv")

    # Modify Date column in chess_data DataFrame to be type Timestamp
    chess_data["Date"] = [pd.to_datetime(chess_date, format="%Y.%m.%d").timestamp() for chess_date in chess_data["Date"]]

    # Create MyElo column in chess_data DataFrame
    find_my_elo(chess_data)

    # For accuracy, consider only points from 2024.08.17 onwards because games are played more consistently then
    new_date_elo = list(zip(chess_data["Date"], chess_data["MyElo"]))

    new_data_date = [date_elo[0] for date_elo in new_date_elo if date_elo[0] > 1723852800]
    new_data_elo = [date_elo[1] for date_elo in new_date_elo if date_elo[0] > 1723852800]

    new_data = {"Date": new_data_date, "MyElo": new_data_elo}
    new_chess_data = pd.DataFrame(new_data)

    # Create model and find information
    X_train, X_test, Y_train, Y_test = train_test_split(new_chess_data["Date"].values.reshape(-1, 1), new_chess_data["MyElo"].values.reshape(-1, 1), test_size = 0.1, train_size = 0.9)
    model = linear_model.LinearRegression().fit(X_train, Y_train)
    respond(f"R square score is {model.score(X_test, Y_test)}", 0)
    respond(f"Coefficient is {model.coef_[0][0]}", 1)
    respond(f"Intercept is {model.intercept_[0]}", 1)

    # Calculate metrics
    Y_pred = model.predict(X_test)
    respond(f"Mean absolute error is {mean_absolute_error(Y_test, Y_pred)}", 1)
    respond(f"Mean squared error is {mean_squared_error(Y_test, Y_pred)}", 1)
    respond(f"Root mean square error is {np.sqrt(mean_squared_error(Y_test, Y_pred))}", 1)

    # Predict Elo
    try:
        predicted_elo = model.coef_[0][0] * pd.to_datetime(prediction_date, format="%Y.%m.%d").timestamp() + model.intercept_[0]
        respond(f"The predicted Elo at {prediction_date} is {predicted_elo}.", 0)
    except Exception as e:
        respond(str(e), 2)
        return

    # Generate plot and line
    new_chess_data["Date"] = pd.to_datetime(new_chess_data["Date"], unit="s")
    plt.scatter(new_chess_data["Date"], new_chess_data["MyElo"])
    x_range = np.arange(new_chess_data["Date"].min().timestamp(), new_chess_data["Date"].max().timestamp() + 1)
    y_range = model.coef_ * x_range + model.intercept_
    plt.plot(pd.to_datetime(x_range, unit="s"), y_range[0], color = "r")

    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m.%d"))
    plt.xticks(rotation=45)

    # Fix labels and title
    plt.xlabel("Date")
    plt.ylabel("Elo")
    plt.title("Elo vs. Date with Linear Regression")

    try:
        plt.savefig("output/plots/linear_regression_predict_elo.png")
    except Exception as e:
        respond(str(e), 2)
        return
    
    plt.show()
    plt.clf()

def find_my_elo(chess_data):
    my_elo = []

    # Reshape columns to create a 1D array per game
    games = np.rot90([chess_data["White"], chess_data["WhiteElo"], chess_data["BlackElo"]], axes=(0, 1))

    # Fill my_elo
    for game in games:
        white_player = game[0]
        white_elo = game[1]
        black_elo = game[2]

        if white_player == chess_name:
            my_elo.append(white_elo)
        else:
            my_elo.append(black_elo)

    my_elo.reverse()

    chess_data["MyElo"] = my_elo

def find_openings():
    # Create a DataFrame with each row as a game and each column as a possible move
    opening_df = create_opening_dataframe()

    # Run apriori algorith from library
    frequent_itemsets = apriori(opening_df.astype("bool"), min_support=1 / len(opening_df), use_colnames=True)

    # Find most common 6-frequent itemsets
    frequent_itemsets = frequent_itemsets[frequent_itemsets["itemsets"].apply(len) == 6]
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
    frequent_itemsets = frequent_itemsets.reset_index(drop=True)

    # Map each itemset to its opening with the most popular move order
    common_openings = []
    support_col = frequent_itemsets["support"]
    moves_col = frequent_itemsets["itemsets"]

    for i in range(0, 5):
        support = support_col[i]
        moves = moves_col[i]

        moves_in_order = get_moves_in_order(moves)

        common_openings.append([moves_in_order, support])

    respond(f"The 5 most common openings (in its most popular move order) are", 0)
    for common_opening in common_openings:
        respond(f"{common_opening[0]} with a play rate of {common_opening[1] * 100:.2f}%", 1)

def create_opening_dataframe():
    # Get list of openings from all PGNs
    chess_data = pd.read_csv("output/csv/chess.csv")
    pgns = chess_data["PGN"]
    openings = [get_opening(pgn) for pgn in pgns]

    # Create set to hold unique moves, list to hold opening moves played
    all_opening_moves = set()
    opening_moves_played = []
    for moves in openings:
        opening_moves = extract_opening_moves(moves)
        opening_moves_played.append(opening_moves)
        all_opening_moves.update(opening_moves)

    all_opening_moves = sorted(list(all_opening_moves))

    # Create a DataFrame with columns for each unique move where 1 means the move was played and 0 means not played
    played_not_played = []
    for opening_moves in opening_moves_played:
        row = [1 if move in opening_moves else 0 for move in all_opening_moves]
        played_not_played.append(row)

    return pd.DataFrame(played_not_played, columns=all_opening_moves)

def extract_opening_moves(moves):
    pattern = r"^1\. (.*?)(?: |$)(.*?)(?: \b2\. |$)(.*?)(?: |$)(.*?)(?: \b3\. |$)(.*?)(?: |$)(.*?)$"
    extracted_opening_moves = re.findall(pattern, moves)[0]

    return extracted_opening_moves

def get_moves_in_order(unordered_moves):
    chess_data = pd.read_csv("output/csv/chess.csv")
    openings = [get_opening(pgn) for pgn in chess_data["PGN"]]

    possible_move_orders = []
    for moves in openings:
        opening_moves = extract_opening_moves(moves)

        # If, for each move in the opening_moves list, it exists in the unordered_moves list, then append opening_moves to possible_move_orders
        if all(move in unordered_moves for move in opening_moves):
            possible_move_orders.append(opening_moves)

    # Find most frequent move order
    popular_move_order = most_frequent(possible_move_orders)

    # Format popular_move_order into PGN
    pgn = "1."
    for index, move in enumerate(popular_move_order):
        if (index % 2 == 0 and index != 0):
            pgn += " " + str(index // 2 + 1) + "."
        pgn += " " + move

    return pgn

def most_frequent(List):
    return max(set(List), key=List.count)

def predict_result(white_elo, black_elo, step_size):
    chess_data = pd.read_csv("output/csv/chess.csv")
    results_enumerated = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}

    # Create model and find information
    X_train, X_test, Y_train, Y_test = train_test_split(chess_data[["WhiteElo", "BlackElo"]].to_numpy(), chess_data["Result"].map(results_enumerated), test_size=0.1, train_size=0.9)

    model_knn = GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors": range(10, 101)})
    model_knn.fit(X_train, Y_train)

    respond(f"Accuracy score is {model_knn.score(X_test, Y_test)}", 0)
    respond(f"Nearest neighbor count is {model_knn.best_params_['n_neighbors']}", 1)

    # Make prediction
    result_prediction = model_knn.predict(pd.DataFrame([{"WhiteElo": white_elo,"BlackElo": black_elo}]).to_numpy())[0]
    results_enumerated_inverted = {v: k for k, v in results_enumerated.items()}
    respond(f"With White Elo of {white_elo} and Black Elo of {black_elo}, the predicted result is {results_enumerated_inverted[result_prediction]}.", 0)

    # Plot scatter
    respond(f"Generating scatter plot with decision boundaries. This may take a few minutes.", 0)
    generate_decision_boundary(chess_data, model_knn, step_size)
    generate_scatter_plot(False)
    plt.title("White Elo vs. Black Elo with Result and Decision Boundaries")
    plt.savefig("output/plots/apriori_predict_result.png")
    respond(f"The scatter plot with decision boundaries is being shown. It is also saved as \"output/plots/apriori_predict_result.png\"", 1)
    plt.show()
    plt.clf()

def generate_decision_boundary(data, clf, step_size):
    x_min, x_max = 0, max(data["WhiteElo"].max(), data["BlackElo"].max()) * 1.1
    y_min, y_max = 0, max(data["WhiteElo"].max(), data["BlackElo"].max()) * 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, z, cmap=ListedColormap(["#558B55", "#FF9055", "#4B3382"]), vmin=0, vmax=3)

def llm_query(user_question):
    tables = query(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'chess';")
    schema_details = "Schema:\n"
    for table in tables:
        table_name = table[0]
        schema_details += f"\nTable: {table_name}\n"
        columns = query(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'chess' AND TABLE_NAME = '{table_name}';")
        for column in columns:
            schema_details += f" {column[0]} ({column[1]})\n"

    client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token)

    user_prompt = f"Given schema details: {schema_details}\nAnswer the user question by providing a SQL query in SQL format: {user_question}"
    
    max_new_tokens=1000
    temperature=0.1

    llm_response = client.text_generation(prompt=user_prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    # Create output file for further details
    respond(f"The full LLM response is saved as \"output/logs/llm_output.txt\".", 0)
    llm_output = open("output/logs/llm_output.txt", "w")
    llm_output.write(llm_response)

    start = llm_response.lower().find("```sql")
    end = llm_response.lower().find(";\n```")
    llm_sql = llm_response[start + 6:end + 1]
    print(f"\nSQL QUERY:{llm_sql}\n\nQUERY RESULTS:")
    try:
        rows = query(llm_sql)
    except Exception as e:
        respond(str(e), 2)
        return
    
    for row in rows:
        print(*row)


def print_menu():
    global initial_run
    if (initial_run):
        print(f"Welcome to Chess DB, featuring over 400 games played on Chess.com played by {chess_name}!")
        initial_run = 0

    print("\nMain Menu - Queries are described below. To select one, enter the corresponding number.\n")

    print("1. Find how many games were played given a date.")
    print("2. Find my win rate given an opening.")
    print("3. Find the top five openings by win rate.")
    print("4. Find how many wins, losses, and draws I have.")
    print("5. Find my highest Elo.\n")

    print("6. Create a scatter plot of game results against the Elo of the two players.")
    print("7. Predict my Elo at a certain date using linear regression.")
    print("8. Find commonly played openings using Apriori.")
    print("9. Use the Elo of two players in a game to predict the result using kNN.")
    print("10. Prompt an LLM to generate a SQL query.\n")

    print("11. Exit program.\n")

    selected_query = input("[INPUT] Enter a number: ")

    try:
        selected_query = int(selected_query)
        if (selected_query > 11 or selected_query < 1):
            respond("Please enter a number between 1 and 11.", 2)
            selected_query = print_menu()
    except:
        respond("Please enter a number.", 2)
        selected_query = print_menu()

    return selected_query

def query(sql):
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def respond(response, response_type):
    if response_type == 0:
        print(f"\n[RESPONSE] {response}")

    if response_type == 1:
        print(f"[RESPONSE] {response}")

    if response_type == 2:
        print(f"\n[EXCEPTION] {response}")

if __name__ == "__main__":
    main()
