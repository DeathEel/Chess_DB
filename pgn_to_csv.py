import os
import re
import csv

game_id = 0

# Adds all .pgn files into a .csv database
def pgn_to_csv():
    # Write header row
    db = open("output/csv/chess.csv", "w", newline='')
    header_row = ("GameID", "Date", "White" , "Black", "Result", "WhiteElo", "BlackElo", "PGN")
    writer = csv.writer(db)
    writer.writerow(header_row)

    # Read every .pgn file in directory
    for file in os.listdir("input"):
        if file.endswith(".pgn"):
            f = open(os.path.join("input", file), "r")
            games = clean_games(f.read()) # clean the file into separate games
            f.close()

            # Append to .csv file
            csv_append(db, games)
    
    db.close()

# Return dict of tags (key) and PGN (value)
def clean_games(file):
    # Each game consists of tags and a PGN. Tags and PGNs are separated by two new lines.
    pattern = r"(?s)(.*?)(?:\n\n|\b$)"    # In English: Capture all characters until two new lines or boundary at end of string
    tags_pgn_split = re.findall(pattern, file)

    # Correlate tags with PGN
    tags_pgn = {}

    # Alternate between tags and PGN except for two empty string captures at the end
    # Remove "\n" from PGN
    # Convert PGN to list
    for index in range(0, len(tags_pgn_split) - 2, 2):
        tags_pgn[tags_pgn_split[index]] = tags_pgn_split[index + 1].replace("\n", " ")

    return tags_pgn

# Appends dict games to .csv file
def csv_append(db, games):
    for tags, pgn in games.items():
        # Capture desired tags
        pattern = r"(?s).*\[Date \"(.*?)\".*\[White \"(.*?)\".*\[Black \"(.*?)\".*\[Result \"(.*?)\".*\[WhiteElo \"(.*?)\".*\[BlackElo \"(.*?)\".*"
        tags_columns = re.findall(pattern, tags)    # Returns a list (matches) of tuples (captures)
        
        # Write data rows
        global game_id
        game_id += 1
        row = tuple([game_id] + list(tags_columns[0]) + [pgn])
        writer = csv.writer(db)
        writer.writerow(row)


if __name__ == "__main__":
    pgn_to_csv()
