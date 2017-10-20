import pandas as pd


def loadLines(fileName, fields):
    """
    Args:
        fileName (str): file to load
        field (set<str>): fields to extract
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    lines = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
        for line in f:
            values = line.split(" +++$+++ ")

            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines

def loadConversations(fileName, lines, fields):
    """
    Args:
        fileName (str): file to load
        field (set<str>): fields to extract
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    conversations = []

    with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
        for line in f:
            values = line.split(" +++$+++ ")

            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]

            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])

            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])

            conversations.append(convObj)

    return conversations

lines = {}
conversations = []

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID",
                              "movieID", "utteranceIDs"]


lines = loadLines("movie_lines.txt", MOVIE_LINES_FIELDS)
conversations = loadConversations(os.path.join("movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)
